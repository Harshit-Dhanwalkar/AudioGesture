// Biquad: Butterworth low-/high-pass magnitudes at, below and above cutoff
#include <cmath>
#include <vector>

#include "audiogesture/audio/Biquad.h"
#include "check.h"

namespace {
constexpr double kSR = 44100.0;

// Steady-state gain of filter for a unit sine at f Hz
double gain(Biquad f, double hz) {
  double in = 0, out = 0;
  const int n = int(kSR);
  for (int i = 0; i < n; ++i) {
    const float x = float(std::sin(6.283185307179586 * hz * i / kSR));
    const float y = f.process(x);
    if (i > n / 2) { // skip transient
      in += double(x) * x;
      out += double(y) * y;
    }
  }
  return std::sqrt(out / in);
}
double db(double g) { return 20.0 * std::log10(g); }
} // namespace

int main() {
  Biquad lp;
  lp.lowpass(1000.0, kSR);
  CHECK_NEAR(db(gain(lp, 100.0)), 0.0, 0.3);   // Pass band
  CHECK_NEAR(db(gain(lp, 1000.0)), -3.0, 0.4); // Butterworth: -3 dB at fc
  CHECK(db(gain(lp, 8000.0)) < -30.0);         // 3 octaves up: ~ -36 dB

  Biquad hp;
  hp.highpass(1000.0, kSR);
  CHECK_NEAR(db(gain(hp, 10000.0)), 0.0, 0.3);
  CHECK_NEAR(db(gain(hp, 1000.0)), -3.0, 0.4);
  CHECK(db(gain(hp, 125.0)) < -30.0);

  // Retuning while running keeps state: no NaN/inf, bounded output
  Biquad sweep;
  float peak = 0;
  for (int i = 0; i < 44100; ++i) {
    if (i % 16 == 0) {
      sweep.lowpass(200.0 * std::pow(100.0, i / 44100.0), kSR);
    }
    const float y = sweep.process(float(std::sin(0.05 * i)));
    peak = std::max(peak, std::fabs(y));
    CHECK(std::isfinite(y));
  }
  CHECK(peak < 2.0f);

  TEST_MAIN_END();
}
