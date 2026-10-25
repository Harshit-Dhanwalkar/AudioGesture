#pragma once
// cava-style audio visualizer drawn straight into OpenCV window
//
//   * spectrum bars : FFT of audio actually leaving speakers (so see effect of
//   stretch / pitch / reverse), log-spaced bands, auto-gain, gravity fall-off,
//   "monstercat" neighbour smoothing, peak caps
//   * waveform strip: overview of whole loaded file with a moving playhead
//
// Class is independent of PortAudio/OpenCV windows: feed it samples, give
// it a canvas

#include <complex>
#include <vector>

#include <opencv2/core.hpp>

class Visualizer {
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

  Visualizer();
  explicit Visualizer(const Config &cfg);

  size_t windowSize() const { return static_cast<size_t>(cfg_.fftSize); }

  // `mono` = windowSize() most recent output samples, oldest first. dt =
  // seconds since last call.
  void update(const float *mono, double sampleRate, double dt);

  void cycleStyle() { style_ = (style_ + 1) % 2; } // bars <-> mirrored bars
  const std::vector<float> &bars() const {
    return disp_;
  } // 0..1, after smoothing (for tests / other UIs)

  // Draw two panels into `canvas`. `song` = whole mono file (for  overview),
  // `playhead` = 0..1 position of read head
  void drawSpectrum(cv::Mat &canvas, const cv::Rect &r);
  void drawWaveform(cv::Mat &canvas, const cv::Rect &r,
                    const std::vector<float> &song, double playhead);

private:
  void buildBands(double sampleRate);
  void buildGradient(cv::Size size, int style);

  Config cfg_;
  int style_ = 0;

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

  // drawing caches
  cv::Mat grad_;
  cv::Size gradSize_;
  int gradStyle_ = -1;
  std::vector<float> waveMin_, waveMax_;
  int waveW_ = 0;
  size_t waveN_ = 0;
};
