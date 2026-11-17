// OSC: wire format byte by byte, per-frame message set, target parsing, and a real UDP round trip
// on loopback
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <cstring>

#include "audiogesture/control/OscOut.h"
#include "check.h"

namespace {
float readFloat(const OscPacket &p, size_t at) {
  std::uint32_t b = (std::uint32_t(p[at]) << 24) | (std::uint32_t(p[at + 1]) << 16) |
                    (std::uint32_t(p[at + 2]) << 8) | p[at + 3];
  float f;
  std::memcpy(&f, &b, 4);
  return f;
}
std::string addressOf(const OscPacket &p) {
  return std::string(reinterpret_cast<const char *>(p.data()));
}
} // namespace

int main() {
  // Spec example bytes: "/a" ,f 1.0  ->  "/a\0\0" ",f\0\0" 3f 80 00 00
  {
    const OscPacket m = oscMessage("/a", {OscArg::ofFloat(1.0f)});
    const std::uint8_t want[] = {'/', 'a', 0, 0, ',', 'f', 0, 0, 0x3f, 0x80, 0, 0};
    CHECK(m.size() == sizeof want && std::memcmp(m.data(), want, sizeof want) == 0);
  }
  // Padding: an address whose length+NUL is already a multiple of 4 gets no extra word; one that is
  // not does
  // Ints are big-endian two's complement
  // No args: just ","
  {
    const OscPacket m = oscMessage("/abc", {OscArg::ofInt(-2)}); // "/abc\0\0\0\0" (8 bytes)
    CHECK(m.size() == 8 + 4 + 4);
    CHECK(m[8] == ',' && m[9] == 'i');
    CHECK(m[12] == 0xff && m[13] == 0xff && m[14] == 0xff && m[15] == 0xfe);
    CHECK(oscMessage("/x", {}).size() == 4 + 4);
    CHECK(oscMessage("/xyz", {OscArg::ofFloat(0), OscArg::ofFloat(0)}).size() % 4 == 0);
    CHECK(oscMessage("/q", {OscArg::ofFloat(-0.5f)}).size() == 12);
    CHECK_NEAR(readFloat(oscMessage("/q", {OscArg::ofFloat(-0.5f)}), 8), -0.5, 0);
  }

  // Per-frame message set
  {
    EngineParams p;
    p.stretch = 2.f;
    p.envL = -1.f;
    p.envR = 0.25f;
    p.freeze = true;
    p.lpHz = 800.f;
    HandState both;
    both.left.valid = both.right.valid = true;
    both.left.x = 0.25f;
    both.right.y = 0.75f;
    const auto msgs = oscGestureFrame(p, both);
    CHECK(msgs.size() == 11);
    for (const auto &m : msgs) {
      CHECK(m.size() % 4 == 0);
      CHECK(addressOf(m).rfind("/audiogesture/", 0) == 0);
    }
    for (const auto &m : msgs) {
      const std::string a = addressOf(m);
      const size_t tags = (a.size() / 4 + 1) * 4; // Start of type-tag string
      const size_t data =
          tags + ((std::strlen(reinterpret_cast<const char *>(m.data()) + tags) + 4) / 4) * 4;
      if (a == "/audiogesture/speed") {
        CHECK_NEAR(readFloat(m, data), 0.5, 1e-6);
      }
      if (a == "/audiogesture/gain/left") {
        CHECK_NEAR(readFloat(m, data), -1.0, 1e-6);
      }
      if (a == "/audiogesture/gain/right") {
        CHECK_NEAR(readFloat(m, data), 0.25, 1e-6);
      }
      if (a == "/audiogesture/lowpass") {
        CHECK_NEAR(readFloat(m, data), 800.0, 1e-3);
      }
      if (a == "/audiogesture/freeze") {
        CHECK(m[data + 3] == 1);
      }
      if (a == "/audiogesture/hands") {
        CHECK(m[data + 3] == 2);
      }
      if (a == "/audiogesture/left") {
        CHECK_NEAR(readFloat(m, data), 0.25, 1e-6);
      }
      if (a == "/audiogesture/right") {
        CHECK_NEAR(readFloat(m, data + 4), 0.75, 1e-6);
      }
    }

    // No hands: no position messages, count 0
    HandState none;
    const auto m0 = oscGestureFrame(EngineParams{}, none);
    CHECK(m0.size() == 9);
    for (const auto &m : m0) {
      CHECK(addressOf(m) != "/audiogesture/left" && addressOf(m) != "/audiogesture/right");
    }
    // Stretch 0 must not produce inf: speed is 0 (message layout: 20-byte address, 4-byte tags)
    EngineParams z;
    z.stretch = 0.f;
    const OscPacket sp = oscGestureFrame(z, none)[0];
    CHECK(addressOf(sp) == "/audiogesture/speed");
    CHECK_NEAR(readFloat(sp, sp.size() - 4), 0.0, 0);
  }

  // Target parsing
  {
    std::string h;
    int port = 0;
    CHECK(parseOscTarget("9000", h, port) && h == "127.0.0.1" && port == 9000);
    CHECK(parseOscTarget(":57120", h, port) && h == "127.0.0.1" && port == 57120);
    CHECK(parseOscTarget("192.168.1.20:8000", h, port) && h == "192.168.1.20" && port == 8000);
    CHECK(parseOscTarget("studio.local:1", h, port) && h == "studio.local" && port == 1);
    CHECK(!parseOscTarget("", h, port));
    CHECK(!parseOscTarget("host:", h, port));
    CHECK(!parseOscTarget("host:0", h, port));
    CHECK(!parseOscTarget("host:65536", h, port));
    CHECK(!parseOscTarget("host:12ab", h, port));
    CHECK(!parseOscTarget("-5", h, port));
  }

  // Real UDP round trip on loopback
  {
    const int rx = ::socket(AF_INET, SOCK_DGRAM, 0);
    CHECK(rx >= 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a.sin_port = 0;
    CHECK(::bind(rx, reinterpret_cast<sockaddr *>(&a), sizeof a) == 0);
    socklen_t len = sizeof a;
    ::getsockname(rx, reinterpret_cast<sockaddr *>(&a), &len);
    timeval tv{2, 0};
    ::setsockopt(rx, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);

    OscSender tx;
    CHECK(!tx.isOpen());
    CHECK(tx.open("127.0.0.1", ntohs(a.sin_port)));
    CHECK(tx.isOpen());
    EngineParams p;
    p.stretch = 4.f;
    HandState none;
    const auto frame = oscGestureFrame(p, none);
    tx.sendAll(frame);

    int got = 0;
    for (size_t i = 0; i < frame.size(); ++i) {
      std::uint8_t buf[256];
      const ssize_t n = ::recv(rx, buf, sizeof buf, 0);
      if (n <= 0) {
        break;
      }
      CHECK(OscPacket(buf, buf + n) == frame[i]); // Loopback keeps order and content
      ++got;
    }
    CHECK(got == int(frame.size()));
    ::close(rx);

    OscSender bad;
    CHECK(!bad.open("no-such-host.invalid", 9000));
    bad.send(frame[0]); // Not open: silently ignored
  }

  TEST_MAIN_END();
}
