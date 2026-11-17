#pragma once
// Settings that should survive between runs: calibration (pose call "1x speed, full volume"),
// segment length, and a few toggles
//
// Stored as plain `key = value` lines in
//   $XDG_CONFIG_HOME/audiogesture/settings.ini   (default ~/.config/audiogesture/)
// WARN: Unknown keys and malformed lines are ignored; every value is clamped to a sane range, so a
// hand-edited or corrupted file can never break the app

#include <string>

struct Settings {
  float restLength = 0.45f; // hand distance (frame heights) that means 1x speed
  float refHeight = 0.45f;  // hand extent (frame heights) that means full amplitude
  float segSec = 4.0f;      // seconds of audio held between the hands
  bool pitchOn = false;
  bool temporalDenoise = true;
  int boostMode = 0; // low-light boost: 0 auto, 1 on, 2 off
  int gridStyle = 0; // 0 bars, 1 mirrored

  void clamp();

  // Where the file lives (creates nothing)
  static std::string defaultPath();

  // load: false if the file is missing/unreadable (defaults are kept)
  bool load(const std::string &path);
  // save: creates the directory if needed
  bool save(const std::string &path) const;
};
