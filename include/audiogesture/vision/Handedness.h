#pragma once
// Left/right assignment for tracked hands
//
// Landmark model emits a "handedness" probability per hand. Which end of
// 0..1 means "left" depends on whether image was mirrored and on how  model was exported, so
// nothing is assumed: while two hands are clearly apart on screen, resolver learns which way
// model's value correlates with screen x (preview is a mirror, so user's left hand is one further
// left on screen). Once that has been seen consistently, handedness can settle cases where screen
// position is ambiguous: hands crossed or touching, or a lone hand reaching over to other half of
// picture Until it is confident it always answers from screen position, i.e. it never does worse
// than before

#include <algorithm>
#include <cmath>

class HandednessResolver {
public:
  static constexpr float kConfident = 0.15f; // |h - 0.5| needed to count as a vote
  static constexpr float kApart = 0.25f;     // x separation (fraction of width) for learning
  static constexpr float kTrust = 0.5f;      // |polarity| needed before it is used

  // h in 0..1 (raw model output, or a smoothed copy); x in 0..1 across the frame.
  static bool confident(float h) { return std::fabs(h - 0.5f) >= kConfident; }

  // Feed every frame in which two hands are tracked.
  void learn(float xa, float ha, float xb, float hb) {
    if (std::fabs(xa - xb) < kApart || !confident(ha) || !confident(hb) ||
        (ha > 0.5f) == (hb > 0.5f)) {
      return; // too close, unsure, or the model thinks both are the same hand
    }

    const bool highIsRight = (ha > hb) == (xa > xb);
    polarity_ += 0.03f * ((highIsRight ? 1.f : -1.f) - polarity_);
    ++votes_;
  }

  bool trusted() const { return votes_ >= 10 && std::fabs(polarity_) >= kTrust; }
  float polarity() const { return polarity_; } // +1: high value = screen-right hand

  // hand at (x, h) user's LEFT hand then `other` = same for second hand if one is tracked
  // Falls back to screen position
  bool isLeft(float x, float h, const float *otherX = nullptr,
              const float *otherH = nullptr) const {
    const bool byScreen = otherX ? x <= *otherX : x < 0.5f;
    if (!trusted() || !confident(h)) {
      return byScreen;
    }
    if (otherX && otherH && confident(*otherH) && (h > 0.5f) == (*otherH > 0.5f)) {
      return byScreen; // model says "same hand twice": no information
    }

    const bool highIsRight = polarity_ > 0.f;
    return (h > 0.5f) != highIsRight; // high value on a right-hand model => not left
  }

private:
  float polarity_ = 0.f;
  int votes_ = 0;
};
