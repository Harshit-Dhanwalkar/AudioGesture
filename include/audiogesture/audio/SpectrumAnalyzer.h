#pragma once
// cava-style spectrum analysis
//
//   * spectrum bars : FFT of audio actually leaving speakers (so see effect of
//   stretch / pitch / reverse), log-spaced bands, auto-gain, gravity fall-off,
//   "monstercat" neighbour smoothing, peak caps
//
// Pure DSP: feed it samples, read bars()/peaks(). Drawing lives in
// ui/SpectrumGrid.h, which maps these bars onto the hand-held grid.

#include <complex>
#include <vector>

class SpectrumAnalyzer {
public:
  struct Config {
    int bars = 48;
    int fftSize =
        4096; // power of two; 4096 @ 44.1 kHz = 93 ms window, 10.8 Hz bins
    float fMin = 40.f;     // Hz, lowest bar edge
    float fMax = 16000.f;  // Hz, highest bar edge
    float rangeDb = 60.f;  // dynamic range mapped onto bar height
    float tiltDbOct = 3.f; // +3 dB/octave: flattens 1/f-like tilt of music
                           // (pink-noise = flat bars)
    float gravity =
        8.f; // fall acceleration in bar-heights / s^2 (higher = snappier)
    float attackMs = 30.f;   // rise time constant
    float monstercat = 1.5f; // neighbour spreading: bar i lifts bar i+-k to
                             // height/monstercat^k (<=1 disables)
  };

  SpectrumAnalyzer();
  explicit SpectrumAnalyzer(const Config &cfg);

  size_t windowSize() const { return static_cast<size_t>(cfg_.fftSize); }

  // `mono` = windowSize() most recent output samples, oldest first. dt =
  // seconds since last call.
  void update(const float *mono, double sampleRate, double dt);

  // 0..1 per band after smoothing / peak-cap positions, for any UI to draw
  const std::vector<float> &bars() const { return disp_; }
  const std::vector<float> &peaks() const { return peak_; }
  int numBars() const { return cfg_.bars; }

private:
  void buildBands(double sampleRate);

  Config cfg_;

  // FFT plan
  std::vector<int> rev_;
  std::vector<std::complex<float>> tw_;
  std::vector<float> win_;
  float winSum_ = 1.f;
  std::vector<std::complex<float>> buf_;
  std::vector<float> amp_; // amplitude spectrum, bins 0..N/2

  // bands
  double bandsFor_ = 0;        // sample rate band table was built for
  std::vector<float> lo_, hi_; // band edges in fractional bins
  std::vector<float> tiltDb_;  // per-band tilt compensation

  // dynamics
  std::vector<float> bar_, vel_, disp_, peak_, peakHold_;
  float refDb_ = -40.f; // running loudness reference (auto sensitivity)
};
