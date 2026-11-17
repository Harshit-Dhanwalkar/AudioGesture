#pragma once
// First-run and help UI: a key/gesture cheat-sheet overlay, a hold-still calibration countdown and
// a hint line while no calibration has been saved

#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

// Hold both hands in your comfortable "1x speed, full volume" pose for `seconds`; losing either
// hand restarts count so a half-seen pose is never stored as calibration
class CalibrationCountdown {
public:
  explicit CalibrationCountdown(double seconds = 3.0) : total_(seconds) {}

  void start() {
    active_ = true;
    held_ = 0;
  }
  void cancel() {
    active_ = false;
    held_ = 0;
  }
  bool active() const { return active_; }
  double remaining() const { return active_ ? total_ - held_ : 0.0; }

  // Call once per frame
  // Returns true exactly once, on frame hold completes
  bool update(double dt, bool bothHands) {
    if (!active_) {
      return false;
    }
    if (!bothHands) {
      held_ = 0;
      return false;
    }

    held_ += dt;
    if (held_ >= total_) {
      active_ = false;
      held_ = 0;
      return true;
    }

    return false;
  }

private:
  double total_, held_ = 0;
  bool active_ = false;
};

// Text of cheat-sheet (one entry per row; "" = blank row)
std::vector<std::string> helpLines();

void drawHelp(cv::Mat &view);
void drawCountdown(cv::Mat &view, const CalibrationCountdown &cd, bool bothHands);
// Single line at bottom: shown until pose has been calibrated once
void drawFirstRunHint(cv::Mat &view);
