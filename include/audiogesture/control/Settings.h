#pragma once
// Settings that should survive between runs: calibration (pose call "1x speed, full volume"),
// segment length, and a few toggles
//
// Stored as plain `key = value` lines in
//   $XDG_CONFIG_HOME/audiogesture/settings.ini   (default ~/.config/audiogesture/)
// WARN: Unknown keys and malformed lines are ignored; every value is clamped to a sane range, so a
// hand-edited or corrupted file can never break app

#include <string>

struct Settings {
  float restLength = 0.45f; // Hand distance (frame heights) that means 1x speed
  float refHeight = 0.45f;  // Hand extent (frame heights) that means full amplitude
  float segSec = 4.0f;      // Seconds of audio held between hands
  bool pitchOn = false;
  bool calibrated = false; // Pose has been calibrated at least once (hides first-run hint)
  bool wsola = true;       // Phase-aligned grains (cleaner time-stretch)
  bool effects = true;     // Pinch freeze + roll filters
  bool temporalDenoise = true;
  int boostMode = 0; // Low-light boost: 0 auto, 1 on, 2 off
  int gridStyle = 0; // 0 bars, 1 mirrored
  bool pan = true;         // Stereo pan from hands' horizontal centre (key x)

  void clamp();

  // Where file lives (creates nothing)
  static std::string defaultPath();

  // Load: false if file is missing/unreadable (defaults are kept)
  bool load(const std::string &path);
  // Save: creates directory if needed
  bool save(const std::string &path) const;
};
