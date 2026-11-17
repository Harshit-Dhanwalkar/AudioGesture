#include "audiogesture/audio/Recorder.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include <sndfile.h>

Recorder::Recorder() : buf_(new float[kCap]) {}

Recorder::~Recorder() { stop(); }

bool Recorder::start(const std::string &path, double sampleRate) {
  if (active()) {
    return false;
  }

  SF_INFO info{};
  info.samplerate = static_cast<int>(sampleRate + 0.5);
  info.channels = 2;
  info.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;
  SNDFILE *f = sf_open(path.c_str(), SFM_WRITE, &info);
  if (!f) {
    std::fprintf(stderr, "cannot record to %s: %s\n", path.c_str(), sf_strerror(nullptr));
    return false;
  }

  file_ = f;
  sr_ = sampleRate;
  w_.store(0);
  r_.store(0);
  frames_.store(0);
  dropped_.store(0);
  quit_.store(false);
  th_ = std::thread(&Recorder::writerLoop, this);
  active_.store(true, std::memory_order_release); // last: push() starts accepting
  return true;
}

void Recorder::stop() {
  if (!th_.joinable()) {
    return;
  }

  active_.store(false, std::memory_order_release); // push() stops accepting
  quit_.store(true, std::memory_order_release);    // writer drains, then exits
  th_.join();
  sf_close(static_cast<SNDFILE *>(file_));
  file_ = nullptr;
}

double Recorder::seconds() const {
  return static_cast<double>(frames_.load(std::memory_order_relaxed)) / sr_;
}

void Recorder::push(const float *x, unsigned long frames) {
  if (!active_.load(std::memory_order_acquire)) {
    return;
  }

  const size_t n = frames * 2;
  const uint64_t w = w_.load(std::memory_order_relaxed);
  const uint64_t r = r_.load(std::memory_order_acquire);
  if (kCap - (w - r) < n) {
    dropped_.fetch_add(frames, std::memory_order_relaxed);
    return;
  }

  const size_t pos = static_cast<size_t>(w & (kCap - 1));
  const size_t first = std::min(n, kCap - pos);
  std::copy(x, x + first, buf_.get() + pos);
  std::copy(x + first, x + n, buf_.get());
  w_.store(w + n, std::memory_order_release);
}

void Recorder::writerLoop() {
  SNDFILE *f = static_cast<SNDFILE *>(file_);
  for (;;) {
    const uint64_t w = w_.load(std::memory_order_acquire);
    const uint64_t r = r_.load(std::memory_order_relaxed);
    if (w == r) {
      if (quit_.load(std::memory_order_acquire) && w_.load(std::memory_order_acquire) == r) {
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      continue;
    }
    const size_t pos = static_cast<size_t>(r & (kCap - 1));
    const size_t n = static_cast<size_t>(w - r);
    const size_t first = std::min(n, kCap - pos);
    sf_write_float(f, buf_.get() + pos, static_cast<sf_count_t>(first));
    if (first < n) {
      sf_write_float(f, buf_.get(), static_cast<sf_count_t>(n - first));
    }
    frames_.fetch_add(n / 2, std::memory_order_relaxed);
    r_.store(w, std::memory_order_release);
  }
}
