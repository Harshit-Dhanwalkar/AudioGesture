// GestureMapper with synthetic hands (no camera, no audio device)
#include "audiogesture/control/GestureMapper.h"
#include "audiogesture/control/Settings.h"
#include "audiogesture/vision/GestureTracker.h"
#include "check.h"

namespace {
// A hand whose wrist and middle fingertip are directly above/below each other
Hand mk(float cx, float wristY, float tipY) {
  Hand h;
  h.valid = h.hasSkeleton = true;
  h.x = cx;
  h.y = 0.5f * (wristY + tipY);
  for (auto &p : h.lm) {
    p = {cx, h.y};
  }
  h.lm[0] = {cx, wristY};
  h.lm[12] = {cx, tipY};

  return h;
}

// A full skeleton: palm length = 0.4 * (wrist-to-tip height)
// pinch = thumb-index distance
// open  = knuckle width, both as ratios of palm length
Hand skel(float cx, float wristY, float tipY, float pinch, float open, float aspect = 4.f / 3.f) {
  Hand h = mk(cx, wristY, tipY);
  const float y9 = wristY + 0.4f * (tipY - wristY), palm = 0.4f * std::fabs(wristY - tipY);
  h.lm[9] = {cx, y9};
  h.lm[5] = {cx - 0.5f * open * palm / aspect, y9};
  h.lm[17] = {cx + 0.5f * open * palm / aspect, y9};
  h.lm[4] = {cx - 0.5f * pinch * palm / aspect, y9 - 0.01f};
  h.lm[8] = {cx + 0.5f * pinch * palm / aspect, y9 - 0.01f};
  return h;
}

void settle(GestureMapper &m, const HandState &hs, int frames = 90) {
  for (int i = 0; i < frames; ++i) {
    m.update(hs, 1.0 / 30.0);
  }
}
} // namespace

int main() {
  HandState hs;
  hs.aspect = 4.f / 3.f;

  // Distance -> Stretch (apart = slower), Proportional to distance
  {
    GestureMapper m;
    hs.left = mk(0.38f, 0.85f, 0.40f);
    hs.right = mk(0.62f, 0.85f, 0.40f);
    settle(m, hs);
    const float close = m.params().stretch;
    hs.left = mk(0.15f, 0.85f, 0.40f);
    hs.right = mk(0.85f, 0.85f, 0.40f);
    settle(m, hs);
    const float apart = m.params().stretch;
    CHECK(apart > 2.f * close);
    CHECK_NEAR(close, 0.24f * 4.f / 3.f / 0.45f, 0.02); // dist / restLength
    CHECK(m.geometry().valid);
  }

  // Edge heights -> Signed gains: rectangle, triangle (fist), inverted, twist
  {
    GestureMapper m;
    hs.left = mk(0.2f, 0.9f, 0.45f);
    hs.right = mk(0.8f, 0.9f, 0.45f); // 0.45 tall = full
    settle(m, hs);
    CHECK_NEAR(m.params().envL, 1.0, 0.02);
    CHECK_NEAR(m.params().envR, 1.0, 0.02);

    hs.right = mk(0.8f, 0.9f, 0.9f); // Fist: no vertical extent
    settle(m, hs);
    CHECK_NEAR(m.params().envR, 0.0, 0.02);
    CHECK(m.params().envL > 0.9f);

    hs.left = mk(0.2f, 0.45f, 0.9f);
    hs.right = mk(0.8f, 0.45f, 0.9f); // Upside down
    settle(m, hs);
    CHECK(m.params().envL < -0.9f && m.params().envR < -0.9f);
    CHECK(m.geometry().eL < 0 && m.geometry().eR < 0);

    hs.left = mk(0.2f, 0.9f, 0.45f); // One up, one down: twist
    settle(m, hs);
    CHECK(m.params().envL > 0.9f && m.params().envR < -0.9f);
  }

  // One hand only: nothing changes, grid hidden
  {
    GestureMapper m;
    hs.left = mk(0.3f, 0.9f, 0.45f);
    hs.right = mk(0.7f, 0.9f, 0.45f);
    settle(m, hs);
    const float s = m.params().stretch;
    hs.left = Hand{};
    settle(m, hs);
    CHECK_NEAR(m.params().stretch, s, 1e-6);
    CHECK(!m.geometry().valid);
  }

  // Direction: a fast sideways sweep of both hands latches; pulling apart does not
  {
    GestureMapper m;
    float lx = 0.5f, rx = 0.8f;
    auto put = [&] {
      hs.left = mk(lx, 0.9f, 0.45f);
      hs.right = mk(rx, 0.9f, 0.45f);
      m.update(hs, 1.0 / 30.0);
    };
    for (int i = 0; i < 20; ++i) {
      put();
    }
    CHECK(m.params().direction > 0);

    // Fast stretch: hands move apart at 0.6 frame widths / s each, centroid still
    for (int i = 0; i < 12; ++i) {
      lx -= 0.02f;
      rx += 0.02f;
      put();
    }
    CHECK(m.params().direction > 0);

    // Sweep both hands left at 0.9 frame widths / s -> reverse
    for (int i = 0; i < 12; ++i) {
      lx -= 0.03f;
      rx -= 0.03f;
      put();
    }
    CHECK(m.params().direction < 0);

    // Back to right -> forward
    for (int i = 0; i < 14; ++i) {
      lx += 0.03f;
      rx += 0.03f;
      put();
    }
    CHECK(m.params().direction > 0);
  }

  // Calibrate: current pose becomes 1x speed / full amplitude
  {
    GestureMapper m;
    hs.left = mk(0.2f, 0.8f, 0.55f);
    hs.right = mk(0.8f, 0.8f, 0.55f); // 0.25 tall
    settle(m, hs);
    m.calibrate(hs);
    settle(m, hs);
    CHECK_NEAR(m.params().stretch, 1.0, 0.03);
    CHECK_NEAR(m.params().envL, 1.0, 0.03);
  }

  // Pitch only when enabled; up = higher
  {
    GestureMapper m;
    hs.left = mk(0.3f, 0.5f, 0.1f);
    hs.right = mk(0.7f, 0.5f, 0.1f); // hands high
    settle(m, hs);
    CHECK_NEAR(m.params().pitchSemis, 0.0, 1e-6);
    m.togglePitch();
    settle(m, hs);
    CHECK(m.params().pitchSemis > 3.f);
  }

  // Short drop-out: grid survives it; a long one hides grid and next sighting snaps to new position
  // instead of gliding from old one
  {
    GestureMapper m;
    hs.left = mk(0.3f, 0.9f, 0.45f);
    hs.right = mk(0.7f, 0.9f, 0.45f);
    settle(m, hs);
    Hand gone;
    HandState miss = hs;
    miss.right = gone;
    for (int i = 0; i < 6; ++i) {
      m.update(miss, 1.0 / 30.0); // 0.2 s
    }
    CHECK(m.geometry().valid);
    CHECK_NEAR(m.geometry().rx, 0.7, 0.01);
    for (int i = 0; i < 12; ++i) {
      m.update(miss, 1.0 / 30.0); // 0.6 s in total
    }
    CHECK(!m.geometry().valid);

    hs.right = mk(0.9f, 0.9f, 0.45f);
    m.update(hs, 1.0 / 30.0);
    CHECK_NEAR(m.geometry().rx, 0.9, 0.01); // snapped, no glide from 0.7
  }

  // Settings round-trip through mapper
  {
    GestureMapper m;
    Settings s;
    s.restLength = 0.7f;
    s.refHeight = 0.2f;
    s.segSec = 8.f;
    s.pitchOn = true;
    m.applySettings(s);
    Settings out;
    m.storeSettings(out);
    CHECK_NEAR(out.restLength, 0.7, 1e-6);
    CHECK_NEAR(out.refHeight, 0.2, 1e-6);
    CHECK_NEAR(out.segSec, 8.0, 1e-6);
    CHECK(out.pitchOn);
  }

  // Effects: pinch freezes (with hysteresis), palm turned edge-on filters
  {
    GestureMapper m;
    auto both = [&](float pinchL, float openL, float pinchR, float openR) {
      hs.left = skel(0.25f, 0.9f, 0.45f, pinchL, openL);
      hs.right = skel(0.75f, 0.9f, 0.45f, pinchR, openR);
      settle(m, hs, 60);
    };
    both(1.2f, 0.7f, 1.2f, 0.7f); // Relaxed open hands: nothing happens
    CHECK(!m.frozen());
    CHECK_NEAR(m.lowpassHz(), 20000.0, 50.0);
    CHECK_NEAR(m.highpassHz(), 20.0, 0.5);

    both(1.2f, 0.7f, 0.1f, 0.7f); // Right pinch
    CHECK(m.frozen() && m.params().freeze);
    both(1.2f, 0.7f, 0.35f, 0.7f); // Between thresholds: still pinched
    CHECK(m.frozen());
    both(1.2f, 0.7f, 0.6f, 0.7f); // Released
    CHECK(!m.frozen());

    both(1.2f, 0.12f, 1.2f, 0.12f);           // Both palms edge-on
    CHECK_NEAR(m.highpassHz(), 1200.0, 30.0); // Left hand -> high-pass up
    CHECK_NEAR(m.lowpassHz(), 400.0, 10.0);   // Right hand -> low-pass down
    both(1.2f, 0.7f, 1.2f, 0.7f);
    CHECK_NEAR(m.lowpassHz(), 20000.0, 50.0);

    // Effects off: bypass even with a pinch and edge-on palms
    m.effectsOn = false;
    both(0.1f, 0.12f, 0.1f, 0.12f);
    CHECK(!m.frozen());
    CHECK_NEAR(m.lowpassHz(), 20000.0, 1.0);
    m.effectsOn = true;

    // Pinch with hand then losing it: sound must not stay frozen
    both(0.1f, 0.7f, 1.2f, 0.7f);
    CHECK(m.frozen());
    HandState gone = hs;
    gone.left = Hand{};
    for (int i = 0; i < 30; ++i) {
      m.update(gone, 1.0 / 30.0); // 1 s, past hold time
    }
    CHECK(!m.frozen());
  }

  // pan: hands' horizontal centre drives stereo pan; off keeps centre; held when a hand leaves
  {
    GestureMapper m;
    hs.left = mk(0.5f, 0.9f, 0.45f);
    hs.right = mk(0.5f, 0.9f, 0.45f); // Centred: pan 0
    settle(m, hs);
    CHECK_NEAR(m.params().pan, 0.0, 0.02);

    hs.left = mk(0.05f, 0.9f, 0.45f);
    hs.right = mk(0.25f, 0.9f, 0.45f); // Both hands well left of centre
    settle(m, hs);
    CHECK(m.params().pan < -0.5f);

    hs.left = mk(0.75f, 0.9f, 0.45f);
    hs.right = mk(0.95f, 0.9f, 0.45f); // Both well right
    settle(m, hs);
    CHECK(m.params().pan > 0.5f);

    // Hard left/right clamps at edge of pose, does not overshoot
    CHECK(m.params().pan <= 1.0f);

    // Off: centre regardless of hand position
    m.panOn = false;
    settle(m, hs); // Still "well right" pose from above
    CHECK_NEAR(m.params().pan, 0.0, 1e-6);
    m.panOn = true;

    // A hand leaving holds last pan rather than snapping to centre
    hs.left = mk(0.75f, 0.9f, 0.45f);
    hs.right = mk(0.95f, 0.9f, 0.45f);
    settle(m, hs);
    const float held = m.params().pan;
    CHECK(held > 0.5f);
    HandState oneGone = hs;
    oneGone.right = Hand{};
    m.update(oneGone, 1.0 / 30.0);
    CHECK_NEAR(m.params().pan, held, 1e-6);
  }

  TEST_MAIN_END();
}
