#pragma once
// One-Euro filter (Casiez, Roussel, Vogel - CHI 2012)
//
// A plain low-pass filter has to pick one trade-off: smooth enough to hide
// jitter when hand is still => laggy when it moves. One-Euro filter makes
// cutoff frequency depend on speed instead:
//
//     cutoff = minCutoff + \beta * |dx/dt|
//
// Slow motion  -> cutoff ~ minCutoff  -> heavy smoothing, no visible jitter
// Fast motion  -> cutoff grows        -> little smoothing, little lag
//
// Each stage is a first-order low-pass with a time-varying coefficient:
//
//     \tau   = 1 / (2 * \pi * cutoff)
//     \alpha = 1 / (1 + \tau/dt)       (dt = time since last sample)
//     y    += \alpha * (x - y)
//
// Tune in units of signal: feed it pixel coordinates, so \beta is "extra Hz of
// cutoff per pixel/second of speed"

#include <algorithm>
#include <cmath>

class OneEuroFilter {
public:
  OneEuroFilter(float minCutoffHz = 1.5f, float beta = 0.02f,
                float derivCutoffHz = 1.0f)
      : minCutoff_(minCutoffHz), beta_(beta), dCutoff_(derivCutoffHz) {}

  void reset() { init_ = false; }

  float operator()(float x, float dt) {
    dt = std::max(dt, 1e-4f);
    if (!init_) {
      init_ = true;
      xHat_ = x;
      dxHat_ = 0.f;
      return x;
    }

    // Speed estimate, itself low-passed so a single noisy sample can't open
    // filter
    const float dx = (x - xHat_) / dt;
    dxHat_ += alpha(dCutoff_, dt) * (dx - dxHat_);

    const float cutoff = minCutoff_ + beta_ * std::fabs(dxHat_);
    xHat_ += alpha(cutoff, dt) * (x - xHat_);
    return xHat_;
  }

private:
  static float alpha(float cutoffHz, float dt) {
    const float tau = 1.f / (6.2831853f * cutoffHz);
    return 1.f / (1.f + tau / dt);
  }

  float minCutoff_, beta_, dCutoff_;
  float xHat_ = 0.f, dxHat_ = 0.f;
  bool init_ = false;
};
