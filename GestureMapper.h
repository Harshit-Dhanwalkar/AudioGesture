#pragma once
// Gesture -> DSP parameter mapping
//
//   distance between hands  -> length of drawn segment -> playback speed.
//        Physical rule: segment holds a fixed stretch of audio, so
//        speed = restLength / length. Pull apart = slower, push toge=
//        faster.
//   height of each hand     -> amplitude at that end of waveform. Height =
//        vertical extent wrist -> middle fingertip. Open hand = loud, fist =
//        silent. Unequal heights give a trapezium, one fist a triangle; gain
//        slides linearly between two edges (engine does same maths).
//   hand upside down        -> that edge's height goes negative -> phase invert
//        sweep both hands sideways together (fast) -> latch reverse / forward
//        (optional, key p) mean height of hands -> pitch +-12 semitones

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <opencv2/opencv.hpp>

#include "AudioEngine.h"
#include "GestureTracker.h"
#include "WaveRig.h"

class GestureMapper {
public:
  // Tunables (keys in main.cpp change some of them)
  float restLength = 0.45f; // hand distance (in frame heights) = 1x speed
  float refHeight = 0.45f;  // hand extent (frame heights) = full amplitude
  float segSec = 4.0f;      // seconds of audio held between hands
  bool pitchOn = false;

  void update(const HandState &hs, double dt, AudioEngine &audio) {
    dt = std::max(dt, 1e-4);
    const float a =
        1.f - std::exp(-float(dt) / 0.06f); // EMA, frame-rate independent

    const bool both = hs.left.valid && hs.right.valid;
    geo_.aspect = hs.aspect;
    if (both) {
      track(L_, hs.left, a);
      track(R_, hs.right, a);

      // Distance in frame heights (aspect-corrected Euclidean)
      const float dx = (R_.x - L_.x) * hs.aspect, dy = R_.y - L_.y;
      const float dist = std::hypot(dx, dy);

      // Speed = rest/length  <=>  stretch factor S = length/rest (S>1 = slower)
      const float S = std::clamp(dist / restLength, 0.25f, 4.f);
      p_.stretch = S;
      p_.grainMs = std::clamp(90.f * std::sqrt(S), 40.f, 250.f);

      // Edge gains: signed vertical extent / reference, soft dead zone near 0
      p_.envL = gainFromExtent(L_.e / refHeight);
      p_.envR = gainFromExtent(R_.e / refHeight);
      p_.gain = 0.8f;
      p_.segSec = segSec;

      // Optional pitch from mean height of hands (up = higher)
      if (pitchOn) {
        const float yMean = 0.5f * (L_.y + R_.y);
        p_.pitchSemis =
            std::clamp((0.5f - yMean) * 2.f * 12.f * 1.4f, -12.f, 12.f);
      } else {
        p_.pitchSemis = 0.f;
      }

      // Sideways sweep of whole waveform -> reverse / forward
      // Centroid velocity vC (frame widths/s) vs change of hand distance vS:
      // pulling hands apart moves centroid at half rate of vS, so requiring
      // |vC| > |vS| rejects stretching and only pure sweeps count
      const float cx = 0.5f * (L_.x + R_.x);
      if (cInit_) {
        const float vC = (cx - cxPrev_) / float(dt),
                    vS = (dist - distPrev_) / float(dt);
        vc_ += (1.f - std::exp(-float(dt) / 0.06f)) * (vC - vc_);
        vs_ += (1.f - std::exp(-float(dt) / 0.06f)) * (vS - vs_);
        if (std::fabs(vc_) > 0.8f && std::fabs(vc_) > 1.2f * std::fabs(vs_)) {
          p_.direction = vc_ > 0 ? 1.f : -1.f;
        }
      }
      cxPrev_ = cx;
      distPrev_ = dist;
      cInit_ = true;

      // Geometry for drawing, same numbers engine gets
      geo_.valid = true;
      geo_.lx = L_.x;
      geo_.ly = L_.y;
      geo_.rx = R_.x;
      geo_.ry = R_.y;
      geo_.eL = p_.envL * refHeight;
      geo_.eR = p_.envR * refHeight;
    } else {
      cInit_ = false; // hold last parameters while a hand is missing
      geo_.valid = false;
      if (hs.left.valid) {
        track(L_, hs.left, a);
      }
      if (hs.right.valid) {
        track(R_, hs.right, a);
      }
    }
    audio.setParams(p_);
  }

  // Remember current pose as "1x speed, full amplitude" (key c)
  void calibrate(const HandState &hs) {
    if (!(hs.left.valid && hs.right.valid)) {
      return;
    }

    const float dx = (R_.x - L_.x) * hs.aspect, dy = R_.y - L_.y;
    restLength = std::clamp(std::hypot(dx, dy), 0.15f, 1.2f);
    const float e = 0.5f * (std::fabs(L_.e) + std::fabs(R_.e));
    refHeight = std::clamp(e, 0.08f, 0.8f);
  }

  void cycleSegment() { segSec = segSec >= 8.f ? 1.f : segSec * 2.f; }
  void togglePitch() { pitchOn = !pitchOn; }
  float direction() const { return p_.direction; }

  void draw(cv::Mat &img, const HandState &hs, const AudioEngine &audio) {
    auto pt = [&](float x, float y) {
      return cv::Point(int(x * img.cols), int(y * img.rows));
    };

    if (hs.left.valid && !hs.left.hasSkeleton) {
      cv::circle(img, pt(L_.x, L_.y), 12, {255, 128, 0}, 2);
    }
    if (hs.right.valid && !hs.right.hasSkeleton) {
      cv::circle(img, pt(R_.x, R_.y), 12, {0, 128, 255}, 2);
    }

    rig_.draw(img, geo_, audio.samples(), audio.fileSampleRate(), segSec,
              audio.playheadNorm(), p_.direction);

    char buf[200];
    std::snprintf(buf, sizeof buf,
                  "speed %.2fx  edges L %+.2f R %+.2f  seg %.0fs  pitch "
                  "%s%+.1fst  dir %+.0f",
                  1.f / p_.stretch, p_.envL, p_.envR, segSec,
                  pitchOn ? "" : "(off) ", p_.pitchSemis, p_.direction);
    cv::putText(img, buf, {10, 24}, cv::FONT_HERSHEY_SIMPLEX, 0.55, {0, 0, 0},
                3, cv::LINE_AA);
    cv::putText(img, buf, {10, 24}, cv::FONT_HERSHEY_SIMPLEX, 0.55,
                {255, 255, 255}, 1, cv::LINE_AA);
    if (!(hs.left.valid && hs.right.valid)) {
      const char *msg = "show both hands: waveform is stretched between them";
      cv::putText(img, msg, {10, img.rows - 12}, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                  {0, 0, 0}, 3, cv::LINE_AA);
      cv::putText(img, msg, {10, img.rows - 12}, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                  {255, 255, 255}, 1, cv::LINE_AA);
    }
  }

private:
  struct Smoothed {
    // centre (frame units)
    float x = 0.5f;
    float y = 0.5f;
    float e = 0.3f; // signed extent
    bool init = false;
  };

  // 0 below ~8% of full, linear above: a relaxed/closed hand reads as "silent"
  static float gainFromExtent(float r) {
    const float s = r < 0 ? -1.f : 1.f, m = std::fabs(r);
    return s * std::clamp((m - 0.08f) / 0.92f, 0.f, 1.f);
  }

  // Edge of rig = vertical segment wrist -> middle fingertip. Its centre is
  // midpoint, its signed height is (wrist.y - tip.y): positive with fingers up,
  // negative upside down, ~0 when hand lies sideways
  static void track(Smoothed &s, const Hand &h, float a) {
    float x = h.x, y = h.y, e = 0.30f;
    if (h.hasSkeleton) {
      x = h.x;
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

  EngineParams p_;
  Smoothed L_, R_;
  RigGeometry geo_;
  WaveRig rig_;
  float cxPrev_ = 0.5f, distPrev_ = 0.f, vc_ = 0.f, vs_ = 0.f;
  bool cInit_ = false;
};
