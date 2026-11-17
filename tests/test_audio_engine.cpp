// AudioEngine, headless: envelope ramp, phase inversion, speed, reverse.
#include <cmath>
#include <memory>
#include <vector>

#include "audiogesture/audio/AudioEngine.h"
#include "check.h"

namespace {
constexpr double kSR = 44100.0;

std::unique_ptr<AudioEngine> makeEngine() {
  // 8 s chirp with a slow tremolo: non-trivial, same loudness everywhere
  std::vector<float> s(size_t(8 * kSR));
  for (size_t n = 0; n < s.size(); ++n) {
    const double t = n / kSR;
    s[n] = float(0.8 * std::sin(6.283185307 * (200 + 100 * t) * t));
  }

  auto e = std::make_unique<AudioEngine>();
  e->loadSamples(std::move(s), kSR);
  e->initOffline(kSR);

  return e;
}

// Render `seconds` and return mono output
std::vector<float> run(AudioEngine &e, double seconds) {
  std::vector<float> out, buf(2 * 256);
  const size_t total = size_t(seconds * kSR);
  while (out.size() < total) {
    e.render(buf.data(), 256);
    for (int i = 0; i < 256; ++i) {
      out.push_back(buf[2 * i]);
    }
  }

  return out;
}

double rms(const std::vector<float> &v, size_t a, size_t b) {
  double s = 0;
  for (size_t i = a; i < b; ++i) {
    s += double(v[i]) * v[i];
  }

  return std::sqrt(s / double(b - a));
}
} // namespace

int main() {
  // 1) Envelope ramp 0 -> 1 across a 4 s segment: loudness grows left to right
  {
    auto ep = makeEngine();
    AudioEngine &e = *ep;
    EngineParams p;
    p.envL = 0.f;
    p.envR = 1.f;
    p.segSec = 4.f;
    e.setParams(p);
    const auto out = run(e, 4.0);
    const size_t q = size_t(kSR);
    const double r0 = rms(out, 0, q), r1 = rms(out, q, 2 * q), r2 = rms(out, 2 * q, 3 * q),
                 r3 = rms(out, 3 * q, 4 * q);
    CHECK(r0 < r1 && r1 < r2 && r2 < r3);
    CHECK(r3 > 3.0 * r0);
    CHECK(e.segmentPhase() >= 0.f && e.segmentPhase() < 1.f);
  }

  // 2) Flat envelope -1 is exact phase inverse of +1
  {
    auto ap = makeEngine(), bp = makeEngine();
    AudioEngine &a = *ap, &b = *bp;
    EngineParams p;
    p.envL = p.envR = 1.f;
    a.setParams(p);
    p.envL = p.envR = -1.f;
    b.setParams(p);
    // let 15 ms smoothers settle (both start at +1), then compare
    run(a, 0.5);
    run(b, 0.5);
    const auto x = run(a, 0.2), y = run(b, 0.2);
    // Different head positions are not an issue: both advanced identically
    double dot = 0, n = 0;
    for (size_t i = 0; i < x.size(); ++i) {
      dot += double(x[i]) * y[i];
      n += double(x[i]) * x[i];
    }
    CHECK_NEAR(dot / n, -1.0, 1e-3);
  }

  // 3) Speed: stretch 2 advances read head half as fast as stretch 1
  {
    auto ap = makeEngine(), bp = makeEngine();
    AudioEngine &a = *ap, &b = *bp;
    EngineParams p;
    p.stretch = 1.f;
    a.setParams(p);
    p.stretch = 2.f;
    b.setParams(p);
    run(a, 1.0);
    run(b, 1.0);
    const double ha = a.playheadNorm(), hb = b.playheadNorm();
    CHECK(ha > 0.0 && hb > 0.0);
    CHECK_NEAR(ha / hb, 2.0, 0.15);
  }

  // 4) Reverse: head moves backwards (wraps below 0 to end of file)
  {
    auto ap = makeEngine();
    AudioEngine &a = *ap;
    EngineParams p;
    p.direction = -1.f;
    a.setParams(p);
    run(a, 0.5);
    CHECK(a.playheadNorm() > 0.5); // wrapped around from 0 -> ~1
  }

  // 5) Stereo is preserved: a left-only file stays left-only, a mono file is centred
  {
    std::vector<float> L(size_t(4 * kSR)), R(L.size(), 0.f);
    for (size_t n = 0; n < L.size(); ++n) {
      L[n] = float(0.6 * std::sin(6.283185307 * 440.0 * n / kSR));
    }
    AudioEngine e;
    e.loadStereo(L, R, kSR);
    e.initOffline(kSR);
    CHECK(e.isStereo());
    std::vector<float> buf(2 * 4096);
    e.render(buf.data(), 4096);
    double el = 0;
    double er = 0;
    for (int i = 0; i < 4096; ++i) {
      el += buf[2 * i] * buf[2 * i];
      er += buf[2 * i + 1] * buf[2 * i + 1];
    }
    CHECK(el > 1.0);
    CHECK(er == 0.0);

    AudioEngine m;
    m.loadSamples(L, kSR);
    m.initOffline(kSR);
    CHECK(!m.isStereo());
    m.render(buf.data(), 4096);
    for (int i = 0; i < 4096; ++i) {
      CHECK(buf[2 * i] == buf[2 * i + 1]);
    }
  }

  TEST_MAIN_END();
}
