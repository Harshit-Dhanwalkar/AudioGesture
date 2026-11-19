#pragma once
// Record and replay hand-tracking sessions
//
// A trace is a plain-text file of HandState frames (timestamp + both hands, with all 21 landmarks
// when present)
// Replaying one drives whole control chain (mapper, audio, UI, OSC) without a camera or neural
// nets, which makes gesture behaviour reproducible: record a tricky moment with `--record-trace`,
// replay it with `--replay` and drop it into tests/data/ so regression tests run it forever
//
//   # audiogesture-trace v1
//   <t seconds> <aspect> <left hand> <right hand>
//   hand := 0 | 1 <x> <y> <score> <hasSkeleton> [42 floats: lm0.x lm0.y ... lm20.y]

#include <chrono>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "audiogesture/vision/GestureTracker.h"

struct TraceFrame {
  double t = 0; // Seconds since start of recording
  HandState state;
};

class TraceWriter {
public:
  bool open(const std::string &path);
  bool isOpen() const { return out_.is_open(); }
  // Skips (returns false) a frame identical to previous one: UI loop polls faster than camera
  // delivers, and duplicates would only bloat file
  bool write(double t, const HandState &hs);
  size_t frames() const { return n_; }

private:
  std::ofstream out_;
  HandState last_;
  bool haveLast_ = false;
  size_t n_ = 0;
};

// Parse a trace; malformed lines are skipped
// Return false if file cannot be opened
bool loadTrace(const std::string &path, std::vector<TraceFrame> &out);

// An IHandSource that plays a trace back
// realtime = honour timestamps
// (default, for app); false = hand over frames as fast as polled (tests)
class ReplaySource : public IHandSource {
public:
  explicit ReplaySource(std::vector<TraceFrame> frames, bool realtime = true, bool loop = true);
  bool poll(HandState &out) override;
  const cv::Mat &frame() const override { return blank_; }
  size_t size() const { return frames_.size(); }

private:
  std::vector<TraceFrame> frames_;
  bool realtime_, loop_;
  size_t next_ = 0;
  std::chrono::steady_clock::time_point start_{};
  double offset_ = 0; // Added to t after a loop restart so time keeps increasing
  cv::Mat blank_;
};
