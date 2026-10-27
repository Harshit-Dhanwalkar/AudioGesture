#include "audiogesture/audio/SpectrumAnalyzer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>


namespace {

constexpr float kPi = 3.14159265358979f;
bool isPow2(int n) { return n > 0 && (n & (n - 1)) == 0; }

} // namespace

SpectrumAnalyzer::SpectrumAnalyzer() : SpectrumAnalyzer(Config{}) {}

SpectrumAnalyzer::SpectrumAnalyzer(const Config &cfg) : cfg_(cfg) {
  if (!isPow2(cfg_.fftSize)) {
    cfg_.fftSize = 4096;
  }

  const int N = cfg_.fftSize;

  // Radix-2 FFT tables: bit-reversal permutation and twiddle factors e^{-2 \pi
  // i k / N}
  rev_.resize(N);
  int bits = 0;
  while ((1 << bits) < N) {
    ++bits;
  }

  for (int i = 0; i < N; ++i) {
    int r = 0;
    for (int b = 0; b < bits; ++b) {
      r |= ((i >> b) & 1) << (bits - 1 - b);
    }

    rev_[i] = r;
  }

  tw_.resize(N / 2);
  for (int k = 0; k < N / 2; ++k) {
    tw_[k] = std::polar(1.f, -2.f * kPi * k / N);
  }

  // Hann window. Its coherent gain is \sum(w)/N = 0.5, so a full-scale sine of
  // amplitude A produces a peak |X[k]| = A * \sum(w) / 2; dividing by that
  // makes 0 dBFS == 1.0
  win_.resize(N);
  winSum_ = 0.f;
  for (int i = 0; i < N; ++i) {
    win_[i] = 0.5f * (1.f - std::cos(2.f * kPi * i / N));
    winSum_ += win_[i];
  }

  buf_.resize(N);
  amp_.assign(N / 2 + 1, 0.f);

  const int B = cfg_.bars;
  bar_.assign(B, 0.f);
  vel_.assign(B, 0.f);
  disp_.assign(B, 0.f);
  peak_.assign(B, 0.f);
  peakHold_.assign(B, 0.f);
}

// Log-spaced band edges, converted to fractional FFT bins for actual sample
// rate
void SpectrumAnalyzer::buildBands(double sr) {
  const int B = cfg_.bars, N = cfg_.fftSize;
  const double binHz = sr / N;
  const double fMax = std::min<double>(cfg_.fMax, 0.45 * sr);
  const double ratio = std::pow(fMax / cfg_.fMin, 1.0 / B);
  lo_.resize(B);
  hi_.resize(B);
  tiltDb_.resize(B);
  for (int i = 0; i < B; ++i) {
    const double f0 = cfg_.fMin * std::pow(ratio, i), f1 = f0 * ratio;
    lo_[i] = float(f0 / binHz);
    hi_[i] = float(f1 / binHz);
    const double fc = std::sqrt(f0 * f1);
    tiltDb_[i] = float(cfg_.tiltDbOct * std::log2(fc / 1000.0));
  }
  bandsFor_ = sr;
}

void SpectrumAnalyzer::update(const float *mono, double sr, double dt) {
  if (sr <= 0)
    return;
  dt = std::clamp(dt, 1e-3, 0.2);
  if (sr != bandsFor_)
    buildBands(sr);
  const int N = cfg_.fftSize, B = cfg_.bars;

  // 1) window + FFT (iterative radix-2 decimation in time)
  for (int i = 0; i < N; ++i)
    buf_[rev_[i]] = {mono[i] * win_[i], 0.f};
  for (int len = 2; len <= N; len <<= 1) {
    const int half = len / 2, step = N / len;
    for (int i = 0; i < N; i += len)
      for (int j = 0; j < half; ++j) {
        const std::complex<float> u = buf_[i + j],
                                  v = buf_[i + j + half] * tw_[j * step];
        buf_[i + j] = u + v;
        buf_[i + j + half] = u - v;
      }
  }
  const float norm = 2.f / winSum_;
  for (int k = 0; k <= N / 2; ++k)
    amp_[k] = std::abs(buf_[k]) * norm;

  // 2) amplitude per band -> dB (+ tilt).  Narrow low bands (less than one bin
  // wide) are read by linear interpolation between bins; wide bands take
  // loudest bin inside them
  std::vector<float> db(B);
  float framePeak = -200.f;
  for (int i = 0; i < B; ++i) {
    float a;
    if (hi_[i] - lo_[i] < 1.f) {
      const float c = 0.5f * (lo_[i] + hi_[i]);
      const int k = std::clamp(int(c), 0, N / 2 - 1);
      const float f = c - k;
      a = amp_[k] * (1.f - f) + amp_[k + 1] * f;
    } else {
      a = 0.f;
      const int k0 = std::clamp(int(std::ceil(lo_[i])), 0, N / 2);
      const int k1 = std::clamp(int(std::floor(hi_[i])), k0, N / 2);
      for (int k = k0; k <= k1; ++k)
        a = std::max(a, amp_[k]);
    }
    db[i] = 20.f * std::log10(std::max(a, 1e-7f)) + tiltDb_[i];
    framePeak = std::max(framePeak, db[i]);
  }

  // 3) auto sensitivity (cava's "autosens"): reference level snaps up to loud
  // peaks and relaxes 3 dB/s when music gets quieter, so any loudness fills
  // display without pegging it. Floor keeps silence from being amplified into
  // noise
  refDb_ = std::max({framePeak, refDb_ - 3.f * float(dt), -45.f});
  const float top = refDb_ + 3.f; // 3 dB headroom
  const float attack = 1.f - std::exp(-float(dt) / (cfg_.attackMs * 1e-3f));

  // 4) rise fast (low-pass), fall with gravity (accelerating, like cava)
  for (int i = 0; i < B; ++i) {
    const float target =
        std::clamp((db[i] - (top - cfg_.rangeDb)) / cfg_.rangeDb, 0.f, 1.f);
    if (target >= bar_[i]) {
      bar_[i] += (target - bar_[i]) * attack;
      vel_[i] = 0.f;
    } else {
      vel_[i] += cfg_.gravity * float(dt);
      bar_[i] = std::max(target, bar_[i] - vel_[i] * float(dt));
    }
  }

  // 5) monstercat smoothing: every bar pulls its neighbours up to height /
  // m^distance, which turns a spiky spectrum into familiar smooth "mountains"
  // Two O(B) sweeps
  disp_ = bar_;
  if (cfg_.monstercat > 1.f) {
    for (int i = 1; i < B; ++i) {
      disp_[i] = std::max(disp_[i], disp_[i - 1] / cfg_.monstercat);
    }
    for (int i = B - 2; i >= 0; --i) {
      disp_[i] = std::max(disp_[i], disp_[i + 1] / cfg_.monstercat);
    }
  }

  // 6) peak caps: hold 0.35 s, then fall
  for (int i = 0; i < B; ++i) {
    if (disp_[i] >= peak_[i]) {
      peak_[i] = disp_[i];
      peakHold_[i] = 0.35f;
    } else if (peakHold_[i] > 0.f)
      peakHold_[i] -= float(dt);
    else
      peak_[i] = std::max(disp_[i], peak_[i] - 0.8f * float(dt));
  }
}
