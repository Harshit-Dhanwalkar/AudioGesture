#include "Visualizer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>

#include <opencv2/imgproc.hpp>

namespace {

constexpr float kPi = 3.14159265358979f;
const cv::Scalar kBg(26, 21, 18); // BGR panel background
const cv::Scalar kBg2(20, 16, 14);

bool isPow2(int n) { return n > 0 && (n & (n - 1)) == 0; }

cv::Scalar lerpColour(const cv::Scalar &a, const cv::Scalar &b, float t) {
  return a * (1.f - t) + b * t;
}

} // namespace

Visualizer::Visualizer() : Visualizer(Config{}) {}

Visualizer::Visualizer(const Config &cfg) : cfg_(cfg) {
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
void Visualizer::buildBands(double sr) {
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

void Visualizer::update(const float *mono, double sr, double dt) {
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

// Vertical gradient (bottom -> top): teal -> violet -> pink.  Pre-rendered once
// per size so each bar is a single copy of a sub-rectangle
void Visualizer::buildGradient(cv::Size size, int style) {
  const cv::Scalar lowC(255, 211, 41), midC(255, 92, 124),
      highC(163, 79, 255); // BGR
  grad_.create(size, CV_8UC3);
  for (int y = 0; y < size.height; ++y) {
    const float t =
        1.f - float(y) / std::max(1, size.height - 1); // 0 at bottom, 1 at top
    const cv::Scalar c = (t < 0.55f)
                             ? lerpColour(lowC, midC, t / 0.55f)
                             : lerpColour(midC, highC, (t - 0.55f) / 0.45f);
    grad_.row(y).setTo(c);
  }

  gradSize_ = size;
  gradStyle_ = style;
}

void Visualizer::drawSpectrum(cv::Mat &img, const cv::Rect &r0) {
  const cv::Rect r = r0 & cv::Rect(0, 0, img.cols, img.rows);
  if (r.width < 16 || r.height < 24) {
    return;
  }
  img(r).setTo(kBg);

  const int B = cfg_.bars;
  const int labelH = 14;
  const int plotH = r.height - labelH;
  const int half = plotH / 2;
  const cv::Size gsz(r.width, style_ == 0 ? plotH : half);
  if (gsz != gradSize_ || style_ != gradStyle_) {
    buildGradient(gsz, style_);
  }

  const float slot = float(r.width) / B;
  const int gap = slot >= 6.f ? 2 : 1;
  const cv::Scalar capColour(235, 235, 235);

  for (int i = 0; i < B; ++i) {
    const int x0 = int(std::lround(i * slot));
    const int bw = int(std::lround((i + 1) * slot)) - x0 - gap;
    if (bw < 1) {
      continue;
    }

    const cv::Rect col(r.x + x0, r.y, bw, plotH);

    if (style_ == 0) {
      const int h = int(disp_[i] * plotH);
      if (h > 0) {
        grad_(cv::Rect(x0, plotH - h, bw, h))
            .copyTo(img(cv::Rect(col.x, r.y + plotH - h, bw, h)));
      }

      const int ph = int(peak_[i] * plotH);
      if (ph > h + 1) {
        cv::rectangle(img, cv::Rect(col.x, r.y + plotH - ph - 1, bw, 2),
                      capColour, cv::FILLED);
      }
    } else {
      const int h = int(disp_[i] * half);
      const int yc = r.y + half;
      if (h > 0) {
        cv::Mat up = grad_(cv::Rect(x0, half - h, bw, h));
        up.copyTo(img(cv::Rect(col.x, yc - h, bw, h)));
        cv::Mat dn;
        cv::flip(up, dn, 0);
        dn.copyTo(img(cv::Rect(col.x, yc, bw, h)));
      }

      const int ph = int(peak_[i] * half);
      if (ph > h + 1) {
        cv::rectangle(img, cv::Rect(col.x, yc - ph - 1, bw, 2), capColour,
                      cv::FILLED);
        cv::rectangle(img, cv::Rect(col.x, yc + ph - 1, bw, 2), capColour,
                      cv::FILLED);
      }
    }
  }

  // Frequency labels on log axis (bars are log-spaced between fMin and
  // effective fMax)
  const double fMax =
      std::min<double>(cfg_.fMax, 0.45 * (bandsFor_ > 0 ? bandsFor_ : 44100.0));
  for (double f : {100.0, 1000.0, 10000.0}) {
    if (f < cfg_.fMin || f > fMax) {
      continue;
    }

    const float t = float(std::log(f / cfg_.fMin) / std::log(fMax / cfg_.fMin));
    char label[32];
    if (f >= 1000) {
      std::snprintf(label, sizeof label, "%dk", int(f / 1000));
    } else {
      std::snprintf(label, sizeof label, "%d", int(f));
    }

    cv::putText(img, label, {r.x + int(t * r.width) - 8, r.y + r.height - 3},
                cv::FONT_HERSHEY_SIMPLEX, 0.36, {150, 140, 130}, 1,
                cv::LINE_AA);
  }
}

void Visualizer::drawWaveform(cv::Mat &img, const cv::Rect &r0,
                              const std::vector<float> &song, double playhead) {
  const cv::Rect r = r0 & cv::Rect(0, 0, img.cols, img.rows);
  if (r.width < 16 || r.height < 8) {
    return;
  }

  img(r).setTo(kBg2);
  if (song.size() < 2) {
    return;
  }

  // Min/max envelope per pixel column; rebuilt only when width or file changes
  if (waveW_ != r.width || waveN_ != song.size()) {
    waveMin_.assign(r.width, 0.f);
    waveMax_.assign(r.width, 0.f);
    float peak = 1e-4f;
    for (int x = 0; x < r.width; ++x) {
      const size_t i0 = song.size() * x / r.width;
      const size_t i1 = std::max(i0 + 1, song.size() * (x + 1) / r.width);
      const size_t stride =
          std::max<size_t>(1, (i1 - i0) / 256); // bound cost on long files
      float lo = 0.f, hi = 0.f;
      for (size_t i = i0; i < i1 && i < song.size(); i += stride) {
        lo = std::min(lo, song[i]);
        hi = std::max(hi, song[i]);
      }

      waveMin_[x] = lo;
      waveMax_[x] = hi;
      peak = std::max({peak, -lo, hi});
    }

    for (int x = 0; x < r.width; ++x) {
      waveMin_[x] /= peak;
      waveMax_[x] /= peak;
    }

    waveW_ = r.width;
    waveN_ = song.size();
  }

  const int mid = r.y + r.height / 2, amp = r.height / 2 - 2;
  const int ph = std::clamp(int(playhead * r.width), 0, r.width - 1);
  for (int x = 0; x < r.width; ++x) {
    const int y0 = mid - int(waveMax_[x] * amp),
              y1 = mid - int(waveMin_[x] * amp);
    const cv::Scalar c = (x <= ph) ? cv::Scalar(205, 170, 120)
                                   : cv::Scalar(120, 100, 80); // played / ahead
    cv::line(img, {r.x + x, y0}, {r.x + x, std::max(y1, y0 + 1)}, c, 1);
  }

  cv::line(img, {r.x + ph, r.y}, {r.x + ph, r.y + r.height - 1},
           {255, 255, 255}, 1);
}
