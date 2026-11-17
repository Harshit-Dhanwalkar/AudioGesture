#pragma once
// Second-order IIR filter (RBJ "Audio EQ Cookbook"), direct form I
//
//   low-pass :  H(s) = 1 / (s^2 + s/Q + 1)
//   high-pass:  H(s) = s^2 / (s^2 + s/Q + 1)
//
// with w0 = 2 * \pi * fc/fs, \alpha = \sin(w0)/(2Q), and for low-pass
//   b0 = (1 - \cos w0)/2, b1 = 1 - \cos w0, b2 = b0
// (high-pass: b0 = (1 + \cos w0)/2, b1 = -(1 + \cos w0), b2 = b0)
//   a0 = 1 + \alpha, a1 = -2 \cos w0, a2 = 1 - \alpha          (all divided by a0)
//
// Q = 1/\sqrt(2) gives flattest pass band (Butterworth): -3 dB at fc and
// 12 dB/octave beyond it
// Coefficients can be changed while running
// State is kept, so a sweeping cutoff does not click

#include <algorithm>
#include <cmath>

class Biquad {
public:
  void lowpass(double fc, double fs, double q = 0.70710678) { design(fc, fs, q, false); }
  void highpass(double fc, double fs, double q = 0.70710678) { design(fc, fs, q, true); }
  void reset() { x1_ = x2_ = y1_ = y2_ = 0.f; }

  float process(float x) {
    const float y = b0_ * x + b1_ * x1_ + b2_ * x2_ - a1_ * y1_ - a2_ * y2_;
    x2_ = x1_;
    x1_ = x;
    y2_ = y1_;
    y1_ = y;

    return y;
  }

private:
  void design(double fc, double fs, double q, bool high) {
    fc = std::clamp(fc, 10.0, 0.49 * fs);
    const double w0 = 6.283185307179586 * fc / fs;
    const double c = std::cos(w0), alpha = std::sin(w0) / (2.0 * q);
    const double a0 = 1.0 + alpha;
    const double b0 = high ? (1.0 + c) / 2.0 : (1.0 - c) / 2.0;
    const double b1 = high ? -(1.0 + c) : (1.0 - c);
    b0_ = float(b0 / a0);
    b1_ = float(b1 / a0);
    b2_ = float(b0 / a0);
    a1_ = float(-2.0 * c / a0);
    a2_ = float((1.0 - alpha) / a0);
  }

  float b0_ = 1.f, b1_ = 0.f, b2_ = 0.f, a1_ = 0.f, a2_ = 0.f;
  float x1_ = 0.f, x2_ = 0.f, y1_ = 0.f, y2_ = 0.f;
};
