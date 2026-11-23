// Stereo pan: equal-power law (no loudness bump/dip across field), centre is unity gain,
// hard left/right silences other channel
#include <cmath>
#include <memory>

#include "audiogesture/audio/AudioEngine.h"
#include "check.h"

namespace {
constexpr double kSR = 48000.0;

struct Levels {
  double l;
  double r;
  bool finite;
};

Levels measure(float pan) {
  AudioEngine e;
  std::vector<float> s(size_t(kSR * 2));
  for (size_t i = 0; i < s.size(); ++i) {
    s[i] = 0.5f * std::sin(6.283185307179586 * 440.0 * double(i) / kSR);
  }
  e.loadSamples(s, kSR);
  e.initOffline(kSR);
  EngineParams p;
  p.pan = pan;
  e.setParams(p);
  std::vector<float> o(2 * 4096);
  for (int b = 0; b < int(1.0 * kSR / 4096); ++b) {
    e.render(o.data(), 4096); // Let pan smoothing settle
  }
  double l = 0;
  double r = 0;
  bool finite = true;
  const int blocks = int(0.5 * kSR / 4096);
  for (int b = 0; b < blocks; ++b) {
    e.render(o.data(), 4096);
    for (int i = 0; i < 4096; ++i) {
      finite = finite && std::isfinite(o[2 * i]) && std::isfinite(o[2 * i + 1]);
      l += double(o[2 * i]) * o[2 * i];
      r += double(o[2 * i + 1]) * o[2 * i + 1];
    }
  }
  return {std::sqrt(l / (blocks * 4096)), std::sqrt(r / (blocks * 4096)), finite};
}
} // namespace

int main() {
  // Centre: both channels equal, at same level as a hard side (equal power, not equal gain)
  const Levels c = measure(0.f);
  CHECK(c.finite);
  CHECK_NEAR(c.l, c.r, 1e-4);

  // Equal-power law: L^2 + R^2 is constant across whole range, not just at ends
  const double p0 = c.l * c.l + c.r * c.r;
  for (float pan : {-1.f, -0.5f, -0.1f, 0.1f, 0.5f, 1.f}) {
    const Levels m = measure(pan);
    CHECK(m.finite);
    CHECK_NEAR(m.l * m.l + m.r * m.r, p0, 0.02 * p0);
  }

  // Hard left: right channel silenced, left unchanged from centre's own-channel level
  {
    const Levels l = measure(-1.f);
    CHECK(l.r < 0.01 * c.l);
    CHECK_NEAR(l.l, c.l / 0.70710678, 0.02 * c.l);
  }
  // Hard right: mirror image
  {
    const Levels r = measure(1.f);
    CHECK(r.l < 0.01 * c.r);
    CHECK_NEAR(r.r, c.r / 0.70710678, 0.02 * c.r);
  }

  // Monotonic: moving further right never makes right channel quieter
  {
    double prevR = 0;
    for (float pan : {-1.f, -0.5f, 0.f, 0.5f, 1.f}) {
      const Levels m = measure(pan);
      CHECK(m.r >= prevR - 1e-6);
      prevR = m.r;
    }
  }

  // Base level with pan stage fully bypassed (panOn = false): whatever engine's own
  // gain-ramp/soft-clip produces for this DC input, with no equal-power attenuation at all.
  auto baseLevel = [&] {
    AudioEngine e;
    std::vector<float> s(size_t(kSR), 0.5f);
    e.loadSamples(s, kSR);
    e.initOffline(kSR);
    EngineParams p;
    p.panOn = false;
    p.pan = 0.8f; // Should be ignored entirely while panOn is false
    e.setParams(p);
    std::vector<float> o(2 * 1024);
    for (int b = 0; b < 20; ++b) {
      e.render(o.data(), 1024);
    }
    for (int i = 0; i < 1024; ++i) {
      CHECK_NEAR(o[2 * i], o[2 * i + 1], 1e-6); // Bypass
    }
    return double(o[0]);
  };
  const double base = baseLevel();
  CHECK(base > 0.1); // sanity: engine is actually producing signal here

  // Default (pan = 0, panOn = true from a default-constructed EngineParams): centred, so both
  // channels still equal, but that is -3 dB equal-power centre relative to bypass level above, not
  // a second bypass
  {
    AudioEngine e;
    std::vector<float> s(size_t(kSR), 0.5f);
    e.loadSamples(s, kSR);
    e.initOffline(kSR);
    e.setParams(EngineParams{});
    std::vector<float> o(2 * 1024);
    for (int b = 0; b < 20; ++b) {
      e.render(o.data(), 1024);
    }
    for (int i = 0; i < 1024; ++i) {
      CHECK_NEAR(o[2 * i], o[2 * i + 1], 1e-6);
    }
    CHECK_NEAR(double(o[0]), base * 0.70710678, 0.01 * base);
  }

  TEST_MAIN_END();
}
