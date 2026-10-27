// SpectrumAnalyzer: a sine lands in the right log-spaced bar; silence is flat.
#include <cmath>
#include <vector>

#include "audiogesture/audio/SpectrumAnalyzer.h"
#include "check.h"

int main() {
  const double sr = 44100.0;
  SpectrumAnalyzer an;
  std::vector<float> x(an.windowSize());

  for (double f : {200.0, 1000.0, 5000.0}) {
    for (size_t i = 0; i < x.size(); ++i) x[i] = float(0.5 * std::sin(6.283185307 * f * i / sr));
    for (int k = 0; k < 20; ++k) an.update(x.data(), sr, 1.0 / 30.0);
    const auto &b = an.bars();
    int arg = 0;
    for (int i = 1; i < int(b.size()); ++i) if (b[i] > b[arg]) arg = i;
    const SpectrumAnalyzer::Config c;
    const double expect = c.bars * std::log(f / c.fMin) / std::log(c.fMax / c.fMin);
    CHECK_NEAR(arg + 0.5, expect, 1.5);
    CHECK(b[arg] > 0.6f);
    // far-away bars are quiet
    const int far = (arg < int(b.size()) / 2) ? int(b.size()) - 1 : 0;
    CHECK(b[far] < 0.2f);
  }

  // silence decays to zero
  std::fill(x.begin(), x.end(), 0.f);
  for (int k = 0; k < 120; ++k) an.update(x.data(), sr, 1.0 / 30.0);
  for (float v : an.bars()) CHECK(v < 0.05f);

  TEST_MAIN_END();
}
