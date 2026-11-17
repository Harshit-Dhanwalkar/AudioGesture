// ThreadedHandSource: newest-wins delivery, frames and states stay paired, end-of-stream
// propagates, destruction joins a busy worker
#include <atomic>
#include <chrono>
#include <thread>

#include "audiogesture/vision/ThreadedSource.h"
#include "check.h"

namespace {
// Produces frame n as an image filled with n and a hand at x = n/1000
class Counter : public IHandSource {
public:
  Counter(int total, int ms) : total_(total), ms_(ms) {}
  bool poll(HandState &out) override {
    if (n_ >= total_) {
      return false;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(ms_));
    ++n_;
    frame_ = cv::Mat(4, 4, CV_8UC1, cv::Scalar(n_ % 250));
    out = HandState{};
    out.left.valid = true;
    out.left.x = n_ / 1000.f;

    return true;
  }

  const cv::Mat &frame() const override { return frame_; }
  int n_ = 0;

private:
  int total_, ms_;
  cv::Mat frame_;
};
} // namespace

int main() {
  // Pairing and ordering: every frame UI sees matches state it came with
  {
    ThreadedHandSource src(std::make_unique<Counter>(60, 1), 5);
    HandState hs;
    int last = 0, polls = 0;
    while (src.poll(hs)) {
      ++polls;
      if (!src.frame().empty()) {
        const int n = src.frame().at<uchar>(0, 0);
        CHECK(hs.left.valid);
        CHECK_NEAR(hs.left.x, n / 1000.f, 1e-6);
        CHECK(n >= last); // never goes backwards
        last = n;
      }
      CHECK(polls < 100000);
    }
    CHECK(last == 60); // Final frame was delivered before end was reported
  }

  // Slow source: UI is not blocked longer than maxWait per poll, and gets newest data repeatedly
  // rather than waiting
  {
    ThreadedHandSource src(std::make_unique<Counter>(8, 40), 5);
    HandState hs;
    int polls = 0;
    const auto t0 = std::chrono::steady_clock::now();
    while (src.poll(hs)) {
      ++polls;
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    CHECK(polls > 8 * 3); // Many more UI iterations than camera frames
    CHECK(s < 1.5);
  }

  // Destroying while worker is mid-stream joins promptly
  {
    const auto t0 = std::chrono::steady_clock::now();
    {
      ThreadedHandSource src(std::make_unique<Counter>(1000000, 5), 5);
      HandState hs;
      src.poll(hs);
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    CHECK(s < 0.5);
  }

  TEST_MAIN_END();
}
