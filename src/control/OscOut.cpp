#include "audiogesture/control/OscOut.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>

namespace {
void pushString(OscPacket &o, const std::string &s) {
  o.insert(o.end(), s.begin(), s.end());
  o.push_back(0); // Terminator, then pad to a multiple of 4
  while (o.size() % 4) {
    o.push_back(0);
  }
}

void pushU32(OscPacket &o, std::uint32_t v) {
  for (int s = 24; s >= 0; s -= 8) {
    o.push_back(std::uint8_t(v >> s));
  }
}
} // namespace

OscPacket oscMessage(const std::string &address, const std::vector<OscArg> &args) {
  OscPacket o;
  pushString(o, address);
  std::string tags = ",";
  for (const OscArg &a : args) {
    tags += a.type == OscArg::Type::Float ? 'f' : 'i';
  }
  pushString(o, tags);
  for (const OscArg &a : args) {
    if (a.type == OscArg::Type::Float) {
      std::uint32_t bits;
      std::memcpy(&bits, &a.f, 4);
      pushU32(o, bits);
    } else {
      pushU32(o, std::uint32_t(a.i));
    }
  }

  return o;
}

std::vector<OscPacket> oscGestureFrame(const EngineParams &p, const HandState &h) {
  using A = OscArg;
  const float speed = p.stretch > 1e-4f ? 1.f / p.stretch : 0.f;
  std::vector<OscPacket> out;
  out.push_back(oscMessage("/audiogesture/speed", {A::ofFloat(speed)}));
  out.push_back(oscMessage("/audiogesture/direction", {A::ofFloat(p.direction)}));
  out.push_back(oscMessage("/audiogesture/gain/left", {A::ofFloat(p.envL)}));
  out.push_back(oscMessage("/audiogesture/gain/right", {A::ofFloat(p.envR)}));
  out.push_back(oscMessage("/audiogesture/pitch", {A::ofFloat(p.pitchSemis)}));
  out.push_back(oscMessage("/audiogesture/freeze", {A::ofInt(p.freeze ? 1 : 0)}));
  out.push_back(oscMessage("/audiogesture/lowpass", {A::ofFloat(p.lpHz)}));
  out.push_back(oscMessage("/audiogesture/highpass", {A::ofFloat(p.hpHz)}));
  out.push_back(
      oscMessage("/audiogesture/hands", {A::ofInt(int(h.left.valid) + int(h.right.valid))}));
  if (h.left.valid) {
    out.push_back(oscMessage("/audiogesture/left", {A::ofFloat(h.left.x), A::ofFloat(h.left.y)}));
  }
  if (h.right.valid) {
    out.push_back(
        oscMessage("/audiogesture/right", {A::ofFloat(h.right.x), A::ofFloat(h.right.y)}));
  }

  return out;
}

bool parseOscTarget(const std::string &text, std::string &host, int &port) {
  std::string h = "127.0.0.1", pt = text;
  const auto colon = text.rfind(':');
  if (colon != std::string::npos) {
    if (colon > 0) {
      h = text.substr(0, colon);
    }
    pt = text.substr(colon + 1);
  }

  if (pt.empty() || pt.size() > 5 || pt.find_first_not_of("0123456789") != std::string::npos) {
    return false;
  }
  const int n = std::atoi(pt.c_str());
  if (n < 1 || n > 65535 || h.empty()) {
    return false;
  }

  host = h;
  port = n;

  return true;
}

struct OscSender::Target {
  sockaddr_storage addr{};
  socklen_t len = 0;
};

OscSender::~OscSender() {
  if (fd_ >= 0) {
    ::close(fd_);
  }
  delete target_;
}

bool OscSender::open(const std::string &host, int port) {
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_DGRAM;
  addrinfo *res = nullptr;
  if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0 || !res) {
    return false;
  }

  const int fd = ::socket(res->ai_family, SOCK_DGRAM, 0);
  if (fd < 0) {
    freeaddrinfo(res);
    return false;
  }

  if (fd_ >= 0) {
    ::close(fd_);
  }
  delete target_;
  target_ = new Target;
  std::memcpy(&target_->addr, res->ai_addr, res->ai_addrlen);
  target_->len = res->ai_addrlen;
  freeaddrinfo(res);
  fd_ = fd;

  return true;
}

void OscSender::send(const OscPacket &p) const {
  if (fd_ >= 0 && target_) {
    (void)::sendto(fd_, p.data(), p.size(), MSG_DONTWAIT,
                   reinterpret_cast<const sockaddr *>(&target_->addr), target_->len);
  }
}
