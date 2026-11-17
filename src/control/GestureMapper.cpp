#include "audiogesture/control/GestureMapper.h"
#include "audiogesture/audio/AudioEngine.h"
#include "audiogesture/vision/GestureTracker.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

float GestureMapper::gainFromExtent(float r) {
  // 0 below ~8% of full, linear above: a relaxed/closed hand reads as "silent"
  const float s = r < 0 ? -1.f : 1.f, m = std::fabs(r);

  return s * std::clamp((m - 0.08f) / 0.92f, 0.f, 1.f);
}

// Edge of grid = vertical segment wrist -> middle fingertip. Its centre is midpoint, its signed
// height is (wrist.y - tip.y): positive with fingers up, negative upside down, ~0 when hand lies
// sideways
void GestureMapper::track(Smoothed &s, const Hand &h, float a) {
  float x = h.x, y = h.y, e = 0.30f;
  if (h.hasSkeleton) {
    y = 0.5f * (h.lm[0].y + h.lm[12].y);
    e = h.lm[0].y - h.lm[12].y;
  }

  if (!s.init) {
    s.x = x;
    s.y = y;
    s.e = e;
    s.init = true;

    return;
  }

  s.x += a * (x - s.x);
  s.y += a * (y - s.y);
  s.e += a * (e - s.e);
}

// Both features are ratios against hand's own palm length (wrist -> middle knuckle),
// so they do not change with how far hand is from camera
//
//  pinch = |thumb tip - index tip| / palm length. A relaxed open hand is ~1+,
//          touching fingertips ~0.1. Schmitt trigger: on below 0.28, off above 0.45
//  open  = |index knuckle - pinky knuckle| / palm length, i.e. apparent palm
//          width. Palm facing camera ~0.7 (measured on a real open hand: 0.69),
//          turned edge-on ~0.1 (it is cos(roll) for a flat hand). 0.12..0.50 is
//          mapped to 0..1, so a normal open hand, even a bit angled, reads exactly 1
//          and nothing is filtered
void GestureMapper::features(const Hand &h, float aspect, float a, bool &pinched, float &open) {
  if (!h.valid || !h.hasSkeleton) {
    return;
  }

  auto d = [&](int i, int j) {
    return std::hypot((h.lm[i].x - h.lm[j].x) * aspect, h.lm[i].y - h.lm[j].y);
  };

  const float palm = std::max(d(0, 9), 1e-4f);
  const float pinch = d(4, 8) / palm;
  pinched = pinched ? pinch < 0.45f : pinch < 0.28f;
  const float o = std::clamp((d(5, 17) / palm - 0.12f) / (0.50f - 0.12f), 0.f, 1.f);
  open += a * (o - open);
}

// Engine parameters from effect state
// Left hand: high-pass 20 Hz -> 1.2 kHz as palm turns edge-on
// Right hand: low-pass 20 kHz -> 400 Hz
// Log-spaced, because pitch perception is
void GestureMapper::applyEffects() {
  if (!effectsOn) {
    p_.freeze = false;
    p_.lpHz = 20000.f;
    p_.hpHz = 20.f;

    return;
  }

  p_.freeze = pinchL_ || pinchR_;
  p_.hpHz = 20.f * std::pow(60.f, 1.f - openL_);
  p_.lpHz = 400.f * std::pow(50.f, openR_);
}

const EngineParams &GestureMapper::update(const HandState &hs, double dt) {
  dt = std::max(dt, 1e-4);
  // EMA smoothing, frame-rate independent: \alpha = 1 - \exp(-dt/\tau)
  const float a = 1.f - std::exp(-float(dt) / 0.06f);

  // Short drop-outs: keep last smoothed position for kHoldSec
  missL_ = hs.left.valid ? 0.f : missL_ + float(dt);
  missR_ = hs.right.valid ? 0.f : missR_ + float(dt);
  if (missL_ >= kHoldSec) {
    L_.init = false; // Next sighting snaps instead of gliding
  }
  if (missR_ >= kHoldSec) {
    R_.init = false;
  }
  if (hs.left.valid) {
    track(L_, hs.left, a);
  }
  if (hs.right.valid) {
    track(R_, hs.right, a);
  }

  const bool both = L_.init && R_.init;
  if (!L_.init) {
    pinchL_ = false; // a hand that is gone cannot keep sound frozen or filtered
    openL_ = 1.f;
  }
  if (!R_.init) {
    pinchR_ = false;
    openR_ = 1.f;
  }

  if (!both) {
    applyEffects();

    // Hold last parameters while a hand is missing
    cInit_ = false;
    geo_.valid = false;

    return p_;
  }

  // Distance in frame heights (aspect-corrected Euclidean)
  const float dx = (R_.x - L_.x) * hs.aspect, dy = R_.y - L_.y;
  const float dist = std::hypot(dx, dy);

  // speed = rest/length  <=>  stretch factor S = length/rest (S > 1 = slower)
  const float S = std::clamp(dist / restLength, 0.25f, 4.f);
  p_.stretch = S;
  p_.grainMs = std::clamp(90.f * std::sqrt(S), 40.f, 250.f);

  // Edge gains: signed vertical extent / reference, soft dead zone near 0
  p_.envL = gainFromExtent(L_.e / refHeight);
  p_.envR = gainFromExtent(R_.e / refHeight);
  p_.gain = 0.8f;
  p_.segSec = segSec;

  features(hs.left, hs.aspect, a, pinchL_, openL_);
  features(hs.right, hs.aspect, a, pinchR_, openR_);
  applyEffects();

  if (pitchOn) {
    const float yMean = 0.5f * (L_.y + R_.y);
    p_.pitchSemis = std::clamp((0.5f - yMean) * 2.f * 12.f * 1.4f, -12.f, 12.f);
  } else {
    p_.pitchSemis = 0.f;
  }

  // Sideways sweep of whole grid -> reverse / forward
  // Centroid velocity vC (frame widths/s) vs change of hand distance vS:
  // pulling hands apart moves centroid at half rate of vS, so requiring |vC| > |vS| rejects
  // stretching and only pure sweeps count
  const float cx = 0.5f * (L_.x + R_.x);
  if (cInit_) {
    const float k = 1.f - std::exp(-float(dt) / 0.06f);
    vc_ += k * ((cx - cxPrev_) / float(dt) - vc_);
    vs_ += k * ((dist - distPrev_) / float(dt) - vs_);
    if (std::fabs(vc_) > 0.8f && std::fabs(vc_) > 1.2f * std::fabs(vs_)) {
      p_.direction = vc_ > 0 ? 1.f : -1.f;
    }
  }

  cxPrev_ = cx;
  distPrev_ = dist;
  cInit_ = true;

  geo_.valid = true;
  geo_.lx = L_.x;
  geo_.ly = L_.y;
  geo_.rx = R_.x;
  geo_.ry = R_.y;
  geo_.eL = p_.envL * refHeight;
  geo_.eR = p_.envR * refHeight;

  return p_;
}

void GestureMapper::calibrate(const HandState &hs) {
  if (!(hs.left.valid && hs.right.valid)) {
    return;
  }

  const float dx = (R_.x - L_.x) * hs.aspect, dy = R_.y - L_.y;
  restLength = std::clamp(std::hypot(dx, dy), 0.15f, 1.2f);
  refHeight = std::clamp(0.5f * (std::fabs(L_.e) + std::fabs(R_.e)), 0.08f, 0.8f);
}

std::string GestureMapper::status() const {
  char buf[200];
  std::snprintf(buf, sizeof buf,
                "speed %.2fx | L %+.2f R %+.2f | seg %.0fs | pitch %s%+.0fst | dir %+.0f",
                1.f / p_.stretch, p_.envL, p_.envR, segSec, pitchOn ? "" : "(off) ", p_.pitchSemis,
                p_.direction);

  std::string out = buf;
  if (p_.freeze) {
    out += " | FREEZE";
  }
  if (p_.lpHz < 15000.f) {
    std::snprintf(buf, sizeof buf, " | LP %.1fk", p_.lpHz / 1000.f);
    out += buf;
  }
  if (p_.hpHz > 30.f) {
    std::snprintf(buf, sizeof buf, " | HP %.0f", p_.hpHz);
    out += buf;
  }

  return out;
}
