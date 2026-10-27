#pragma once
// Gesture -> DSP parameter mapping: "hold the sound between your hands".
//
//   distance between hands  -> length of the grid -> playback speed.
//        Physical rule: the grid holds a fixed stretch of audio, so
//        speed = restLength / length. Pull apart = slower, push together = faster.
//   height of each hand     -> amplitude at that end of the grid. Height =
//        vertical extent wrist -> middle fingertip. Open hand = loud, fist = silent.
//        Unequal heights give a trapezium, one fist a triangle; the gain slides
//        linearly between the two edges (the engine does the same maths).
//   hand upside down        -> that edge's height goes negative -> phase invert
//   sweep both hands sideways together (fast) -> latch reverse / forward
//   (optional) mean height of the hands -> pitch +-12 semitones
//
// Pure logic: no drawing, no audio device. update() turns a HandState into an
// EngineParams; main.cpp forwards it to the engine.

#include <string>

#include "audiogesture/audio/AudioEngine.h"
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

  void cycleSegment() { segSec = segSec >= 8.f ? 1.f : segSec * 2.f; }
  void togglePitch() { pitchOn = !pitchOn; }

  const EngineParams &params() const { return p_; }
  // Grid for the UI; same numbers the engine gets, so drawing == hearing
  const RigGeometry &geometry() const { return geo_; }
  std::string status() const; // one-line HUD text

private:
  struct Smoothed {
    float x = 0.5f, y = 0.5f, e = 0.3f; // centre (frame units) and signed extent
    bool init = false;
  };
  static void track(Smoothed &s, const Hand &h, float a);
  static float gainFromExtent(float r);

  EngineParams p_;
  Smoothed L_, R_;
  RigGeometry geo_;
  float cxPrev_ = 0.5f, distPrev_ = 0.f, vc_ = 0.f, vs_ = 0.f;
  bool cInit_ = false;
};
