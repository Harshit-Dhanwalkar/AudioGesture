#pragma once
// Records engine's stereo output to a WAV file without ever blocking audio thread: push() copies
// into a single-producer / single-consumer ring and a writer thread drains it to libsndfile. If
// disk stalls long enough to fill ring (~6 s), frames are dropped and counted rather than waiting

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>

class Recorder {
public:
  Recorder();
  ~Recorder();
  Recorder(const Recorder &) = delete;
  Recorder &operator=(const Recorder &) = delete;

  // Opens `path` (32-bit float stereo WAV) and starts writer thread
  bool start(const std::string &path, double sampleRate);
  // Flushes everything already pushed, closes file
  // Safe to call twice
  void stop();

  bool active() const { return active_.load(std::memory_order_relaxed); }
  double seconds() const;
  size_t dropped() const { return static_cast<size_t>(dropped_.load(std::memory_order_relaxed)); }

  // Audio thread: interleaved stereo, `frames` frames
  // Wait-free
  void push(const float *interleaved, unsigned long frames);

private:
  void writerLoop();

  static constexpr size_t kCap = size_t(1) << 19; // floats; power of two
  std::unique_ptr<float[]> buf_;
  std::atomic<uint64_t> w_{0}, r_{0};
  std::atomic<bool> active_{false}, quit_{false};
  std::atomic<uint64_t> frames_{0}, dropped_{0};
  double sr_ = 44100.0;
  std::thread th_;
  void *file_ = nullptr; // SNDFILE*
};
