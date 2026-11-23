// Stretch quality, measured: a steady tone stretched by plain granular overlap-add has its level
// ripple (grains overlap out of phase and cancel); waveform-similarity placement (WSOLA) must keep
// it as steady as unstretched tone
// "Flutter" = std/mean of RMS in 5 ms windows

#include <cmath>
#include <memory>

#include "audiogesture/audio/AudioEngine.h"
#include "check.h"

namespace {
constexpr double kSR = 48000.0;
constexpr double kTwoPi = 6.283185307179586;

std::vector<float> tone(int kind) {
  std::vector<float> s(size_t(kSR * 10));
  for (size_t i = 0; i < s.size(); ++i) {
    const double t = double(i) / kSR;
    double v = 0;
    if (kind == 0) {
      v = 0.6 * std::sin(kTwoPi * 440 * t); // pure tone
    } else {                                // voiced-like: 130 Hz pulse train plus its fundamental
      const double ph = std::fmod(130 * t, 1.0);
      v = 0.5 * (ph < 0.05 ? 1.0 : -0.05) + 0.3 * std::sin(kTwoPi * 130 * t);
    }
    s[i] = float(v);
  }
  return s;
}

struct Result {
  double mean;
  double flutter;
  bool finite;
};

Result measure(int kind, float stretch, bool wsola, float dir = 1.f, float pitch = 0.f) {
  AudioEngine e;
  e.loadSamples(tone(kind), kSR);
  e.initOffline(kSR);
  EngineParams p;
  p.stretch = stretch;
  p.wsola = wsola;
  p.gain = 0.5f;
  p.direction = dir;
  p.pitchSemis = pitch;
  e.setParams(p);
  std::vector<float> o(2 * 4096), mono;
  for (int b = 0; b < int(3.0 * kSR / 4096); ++b) {
    e.render(o.data(), 4096);
    for (int i = 0; i < 4096; ++i) {
      mono.push_back(o[2 * i]);
    }
  }
  Result r{0, 0, true};
  const int W = int(0.005 * kSR);
  std::vector<double> rms;
  for (size_t i = size_t(0.5 * kSR); i + W < mono.size(); i += W) {
    double a = 0;
    for (int k = 0; k < W; ++k) {
      r.finite = r.finite && std::isfinite(mono[i + k]);
      a += double(mono[i + k]) * mono[i + k];
    }
    rms.push_back(std::sqrt(a / W));
  }
  for (double x : rms) {
    r.mean += x;
  }
  r.mean /= double(rms.size());
  double v = 0;
  for (double x : rms) {
    v += (x - r.mean) * (x - r.mean);
  }
  r.flutter = std::sqrt(v / double(rms.size())) / std::max(r.mean, 1e-9);
  return r;
}
} // namespace

int main() {
  // Reference: unstretched tone (head moves at natural speed: no overlap problem)
  const Result ref0 = measure(0, 1.0f, true);
  CHECK(ref0.flutter < 0.04);

  for (float st : {0.6f, 1.5f, 2.5f, 4.0f}) {
    const Result plain = measure(0, st, false), w = measure(0, st, true);
    std::printf("tone  stretch %.1f: flutter plain %.3f -> wsola %.3f, level %.3f -> %.3f\n", st,
                plain.flutter, w.flutter, plain.mean, w.mean);
    CHECK(w.finite && plain.finite);
    CHECK(w.flutter < 0.04);                         // as steady as unstretched
    CHECK(w.flutter <= plain.flutter + 1e-9);        // never worse
    CHECK_NEAR(w.mean, ref0.mean, 0.03 * ref0.mean); // level not eaten by cancellation
  }
  CHECK(measure(0, 0.6f, false).flutter > 0.15);

  // Voiced-like signal: clearly steadier at stretches where plain overlap-add cancels
  {
    const Result ref = measure(1, 1.0f, true);
    for (float st : {1.5f, 2.5f}) {
      const Result plain = measure(1, st, false), w = measure(1, st, true);
      std::printf("voice stretch %.1f: flutter plain %.3f -> wsola %.3f\n", st, plain.flutter,
                  w.flutter);
      CHECK(w.flutter < 0.6 * plain.flutter);
      CHECK(w.flutter < ref.flutter + 0.03);
    }
  }

  // Reverse and pitch-shifted playback stay steady and finite too
  {
    const Result r = measure(0, 2.0f, true, -1.f);
    CHECK(r.finite && r.flutter < 0.05);
    const Result up = measure(0, 2.0f, true, 1.f, 5.f);
    CHECK(up.finite && up.flutter < 0.08);
  }

  // Silence in, silence out (no NaN from normalisation)
  {
    AudioEngine e;
    e.loadSamples(std::vector<float>(size_t(kSR * 2), 0.f), kSR);
    e.initOffline(kSR);
    EngineParams p;
    p.stretch = 2.f;
    e.setParams(p);
    std::vector<float> o(2 * 4096);
    for (int b = 0; b < 20; ++b) {
      e.render(o.data(), 4096);
      for (float x : o) {
        CHECK(x == 0.f);
      }
    }
  }

  TEST_MAIN_END();
}
