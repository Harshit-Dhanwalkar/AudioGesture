// Latency and CPU budget: how long until a gesture is audible, how much of audio callback's time
// engine uses, and how stale hand data can be when UI reads it
// Bounds are generous (CI machines are noisy); measured values are printed so regressions are
// visible

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <thread>

#include "audiogesture/audio/AudioEngine.h"
#include "audiogesture/vision/ThreadedSource.h"
#include "check.h"

namespace {
constexpr double kSR = 48000.0;
using Clock = std::chrono::steady_clock;

std::unique_ptr<AudioEngine> makeEngine() {
  std::vector<float> s(size_t(kSR * 8));
  for (size_t i = 0; i < s.size(); ++i) {
    s[i] = 0.5f * std::sin(6.2831853f * 440.f * float(i) / float(kSR)) +
           0.2f * std::sin(6.2831853f * 3000.f * float(i) / float(kSR));
  }
  auto e = std::make_unique<AudioEngine>();
  e->loadSamples(std::move(s), kSR);
  e->initOffline(kSR);
  return e;
}

double rms(const float *p, int n) {
  double a = 0;
  for (int i = 0; i < n; ++i)
    a += double(p[i]) * p[i];
  return std::sqrt(a / n);
}
} // namespace

int main() {
  // 1. Real-time factor: engine with everything switched on (stretch, pitch, filters, freeze on and
  // off) must use a small fraction of audio callback's time budget
  {
    auto e = makeEngine();
    EngineParams p;
    p.stretch = 1.7f;
    p.pitchSemis = 3.f;
    p.lpHz = 2000.f;
    p.hpHz = 200.f;
    p.envL = -0.5f;
    p.envR = 1.f;
    e->setParams(p);
    std::vector<float> buf(2 * 256); // interleaved stereo
    const int blocks = int(10.0 * kSR / 256);
    const auto t0 = Clock::now();
    for (int b = 0; b < blocks; ++b) {
      if (b % 400 == 0) {
        p.freeze = !p.freeze;
        e->setParams(p);
      }
      e->render(buf.data(), 256);
    }
    const double took = std::chrono::duration<double>(Clock::now() - t0).count();
    const double rtf = took / 10.0;
    std::printf("engine real-time factor %.4f (%.1f%% of the callback budget)\n", rtf, 100 * rtf);
    CHECK(rtf < 0.5);
  }

  // 2. Gesture -> sound: after gain targets drop to zero, how many ms until output has fallen below
  // 10 % (and, for a freeze, until read head has stopped)?
  {
    auto e = makeEngine();
    EngineParams p;
    e->setParams(p);
    std::vector<float> buf(2 * 64);
    for (int i = 0; i < int(0.5 * kSR / 64); ++i)
      e->render(buf.data(), 64); // reach steady state
    double before = 0;
    for (int i = 0; i < 50; ++i) {
      e->render(buf.data(), 64);
      before = std::max(before, rms(buf.data(), 2 * 64));
    }
    CHECK(before > 0.05);

    p.envL = p.envR = 0.f;
    e->setParams(p);
    int blocks = 0;
    while (blocks < 2000) {
      e->render(buf.data(), 64);
      ++blocks;
      if (rms(buf.data(), 2 * 64) < 0.1 * before)
        break;
    }
    const double ms = 1000.0 * blocks * 64 / kSR;
    std::printf("gain gesture -> -20 dB: %.1f ms\n", ms);
    CHECK(ms < 150.0);
    CHECK(ms > 0.5); // It is smoothed: an instant step would click
  }

  // 3. Hand data staleness through worker thread: time from worker publishing a frame to UI loop
  // receiving it
  {
    struct Stamped : IHandSource {
      bool poll(HandState &out) override {
        if (n >= 200)
          return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        out = HandState{};
        out.left.valid = true;
        out.left.x = float(n) / 1000.f;
        stamp[n] = Clock::now();
        ++n;
        return true;
      }
      const cv::Mat &frame() const override { return f; }
      int n = 0;
      Clock::time_point stamp[200];
      cv::Mat f = cv::Mat(2, 2, CV_8UC1, cv::Scalar(1));
    };
    auto inner = std::make_unique<Stamped>();
    Stamped *raw = inner.get();
    ThreadedHandSource src(std::move(inner), 16);
    HandState hs;
    std::vector<double> lat;
    int lastSeen = -1;
    while (src.poll(hs)) {
      const int n = int(std::lround(hs.left.x * 1000.f));
      if (hs.left.valid && n != lastSeen) {
        lastSeen = n;
        // stamp[n] was written before frame was published, under hand-over mutex
        lat.push_back(
            std::chrono::duration<double, std::milli>(Clock::now() - raw->stamp[n]).count());
      }
    }
    CHECK(lat.size() > 100);
    std::sort(lat.begin(), lat.end());
    const double med = lat[lat.size() / 2], p95 = lat[size_t(lat.size() * 0.95)];
    std::printf("tracking -> UI hand-over: median %.1f ms, p95 %.1f ms\n", med, p95);
    CHECK(med < 30.0);
    CHECK(p95 < 80.0);
  }

  TEST_MAIN_END();
}
