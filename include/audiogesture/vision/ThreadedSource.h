#pragma once
// Runs any IHandSource on its own thread
//
// Camera read + low-light enhancement + two neural-net passes take 20-60 ms;
// UI thread (drawing, spectrum, audio parameter smoothing, key handling) should not stall behind
// them
// Worker polls wrapped source flat out and publishes newest {HandState, frame};
// poll() hands over whatever is newest, waiting at most `maxWaitMs` for a fresh one so UI loop
// idles instead of spinning when camera is slower side
//
// Wrapped source must not be touched from other threads while this runs
// (except through members it documents as thread-safe)

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

#include "audiogesture/vision/GestureTracker.h"

class ThreadedHandSource : public IHandSource {
public:
  explicit ThreadedHandSource(std::unique_ptr<IHandSource> inner, int maxWaitMs = 16)
      : inner_(std::move(inner)), maxWaitMs_(maxWaitMs) {
    worker_ = std::thread([this] { run(); });
  }

  ~ThreadedHandSource() override {
    stop_ = true;
    cv_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  ThreadedHandSource(const ThreadedHandSource &) = delete;
  ThreadedHandSource &operator=(const ThreadedHandSource &) = delete;

  // False once wrapped source has ended and its last frame was delivered
  bool poll(HandState &out) override {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait_for(lk, std::chrono::milliseconds(maxWaitMs_),
                 [&] { return seq_ != seenSeq_ || ended_; });
    if (seq_ == 0) {
      return !ended_; // Nothing yet (camera warming up): keep UI alive
    }

    if (seq_ != seenSeq_) {
      seenSeq_ = seq_;
      frame_ = pubFrame_; // Header copy: worker allocates a fresh buffer each time
      ++delivered_;
    } else if (ended_) {
      return false;
    }

    out = pubState_;
    return true;
  }

  const cv::Mat &frame() const override { return frame_; }

  unsigned long framesDelivered() const { return delivered_; } // New frames seen by poll()

private:
  void run() {
    HandState st;
    while (!stop_) {
      if (!inner_->poll(st)) {
        break;
      }

      cv::Mat f = inner_->frame().clone(); // Inner reuses its buffer next iteration
      std::lock_guard<std::mutex> lk(mu_);
      pubState_ = st;
      pubFrame_ = std::move(f);
      ++seq_;
      cv_.notify_all();
    }

    std::lock_guard<std::mutex> lk(mu_);
    ended_ = true;
    cv_.notify_all();
  }

  std::unique_ptr<IHandSource> inner_;
  int maxWaitMs_;
  std::thread worker_;
  std::atomic<bool> stop_{false};

  std::mutex mu_;
  std::condition_variable cv_;
  HandState pubState_;
  cv::Mat pubFrame_;
  unsigned long seq_ = 0, seenSeq_ = 0;
  bool ended_ = false;

  cv::Mat frame_; // UI-thread copy returned by frame()
  unsigned long delivered_ = 0;
};
