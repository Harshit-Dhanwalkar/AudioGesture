#pragma once
// OSC (Open Sound Control) output over UDP: publish what hands are doing 
//
// Messages sent each frame (all under /audiogesture):
//   /speed f        playback speed, 1 = calibrated rest pose (1 / stretch)
//   /direction f    +1 forward ... -1 reverse
//   /gain/left f    signed edge gain under left hand  (-1 .. +1)
//   /gain/right f   signed edge gain under right hand
//   /pitch f        semitones (0 unless pitch mode is on)
//   /freeze i       1 while a hand pinches
//   /lowpass f      cut-off in Hz (20000 = open)
//   /highpass f     cut-off in Hz (20 = open)
//   /hands i        number of hands tracked (0..2)
//   /left f f       left hand palm centre x, y  (0..1, y down)  - only while tracked
//   /right f f      right hand palm centre x, y                 - only while tracked

#include <cstdint>
#include <string>
#include <vector>

#include "audiogesture/audio/AudioEngine.h"
#include "audiogesture/vision/GestureTracker.h"

using OscPacket = std::vector<std::uint8_t>;

struct OscArg {
  enum class Type { Float, Int } type = Type::Float;
  float f = 0.f;
  std::int32_t i = 0;
  static OscArg ofFloat(float v) {
    OscArg a;
    a.type = Type::Float;
    a.f = v;
    return a;
  }
  static OscArg ofInt(std::int32_t v) {
    OscArg a;
    a.type = Type::Int;
    a.i = v;
    return a;
  }
};

// One OSC message: address, type tags, big-endian arguments, all padded to 4 bytes
OscPacket oscMessage(const std::string &address, const std::vector<OscArg> &args);

// Everything described above, one packet per message
std::vector<OscPacket> oscGestureFrame(const EngineParams &p, const HandState &hands);

// "host:port", ":port" or "port"  ->  host (default 127.0.0.1) + port
// Return false if malformed
bool parseOscTarget(const std::string &text, std::string &host, int &port);

class OscSender {
public:
  OscSender() = default;
  ~OscSender();
  OscSender(const OscSender &) = delete;
  OscSender &operator=(const OscSender &) = delete;

  bool open(const std::string &host, int port); // Resolves and creates socket
  bool isOpen() const { return fd_ >= 0; }
  // Fire and forget; a full buffer or an unreachable receiver is not an error worth stopping for
  void send(const OscPacket &p) const;
  void sendAll(const std::vector<OscPacket> &ps) const {
    for (const auto &p : ps) {
      send(p);
    }
  }

private:
  int fd_ = -1;
  struct Target;
  Target *target_ = nullptr;
};
