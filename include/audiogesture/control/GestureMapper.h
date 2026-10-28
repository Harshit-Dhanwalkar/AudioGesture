#pragma once
// Gesture -> DSP parameter mapping: "hold sound between your hands"
//
//   distance between hands  -> length of grid -> playback speed
//        Physical rule: grid holds a fixed stretch of audio, so
//        speed = restLength / length. Pull apart = slower, push together = faster
//   height of each hand     -> amplitude at that end of grid. Height =
//        vertical extent wrist -> middle fingertip. Open hand = loud, fist = silent
//        Unequal heights give a trapezium, one fist a triangle; gain slides
//        linearly between two edges (engine does same maths)
//   hand upside down        -> that edge's height goes negative -> phase invert
//   sweep both hands sideways together (fast) -> latch reverse / forward (optional) mean height of
//   hands -> pitch \pm 12 semitones
//
// Pure logic: no drawing, no audio device. update() turns a HandState into an
// EngineParams; main.cpp forwards it to engine

#include <string>

#include "audiogesture/audio/AudioEngine.h"
#include "audiogesture/control/Settings.h"
#include "audiogesture/ui/SpectrumGrid.h"
#include "audiogesture/vision/GestureTracker.h"

class GestureMapper {
public:
  // Tunables (keys in main.cpp change some of them)
  float restLength = 0.45f; // hand distance (in frame heights) = 1x speed
  float refHeight = 0.45f;  // hand extent (frame heights) = full amplitude
  float segSec = 4.0f;      // seconds of audio held between the hands
  bool pitchOn = false;

  // Returns the parameters to hand to AudioEngine::setParams()
  const EngineParams &update(const HandState &hs, double dt);

  // Remember the current pose as "1x speed, full amplitude"
  void calibrate(const HandState &hs);

  // A hand that disappears for less than this keeps its last position, so one missed detection does
  // not make grid (and its audio) flicker
  static constexpr float kHoldSec = 0.4f;

  void applySettings(const Settings &st) {
    restLength = st.restLength;
    refHeight = st.refHeight;
    segSec = st.segSec;
    pitchOn = st.pitchOn;
  }
  void storeSettings(Settings &st) const {
    st.restLength = restLength;
    st.refHeight = refHeight;
    st.segSec = segSec;
    st.pitchOn = pitchOn;
  }

  void cycleSegment() { segSec = segSec >= 8.f ? 1.f : segSec * 2.f; }
  void togglePitch() { pitchOn = !pitchOn; }

  const EngineParams &params() const { return p_; }
  // Grid for UI; same numbers engine gets, so drawing == hearing
  const RigGeometry &geometry() const { return geo_; }
  std::string status() const; // one-line HUD text

private:
  struct Smoothed {
    float x = 0.5f;
    float y = 0.5f;
    float e = 0.3f; // centre (frame units) and signed extent
    bool init = false;
  };

  static void track(Smoothed &s, const Hand &h, float a);
  static float gainFromExtent(float r);

  EngineParams p_;
  Smoothed L_, R_;
  RigGeometry geo_;
  float cxPrev_ = 0.5f;
  float distPrev_ = 0.f;
  float vc_ = 0.f;
  float vs_ = 0.f;
  bool cInit_ = false;
  float missL_ = 1e3f;
  float missR_ = 1e3f; // seconds since each hand was last seen
};
