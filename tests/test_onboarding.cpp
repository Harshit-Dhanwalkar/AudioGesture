// Calibration countdown, help overlay, file picker (with a fake command runner).
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

#include "audiogesture/ui/FilePicker.h"
#include "audiogesture/ui/Onboarding.h"
#include "check.h"

namespace {
int changedPixels(const cv::Mat &a, const cv::Mat &b) {
  cv::Mat d;
  cv::absdiff(a, b, d);
  cv::cvtColor(d, d, cv::COLOR_BGR2GRAY);

  return cv::countNonZero(d);
}
} // namespace

int main() {
  // Countdown: completes after holding, exactly once, and only with both hands
  {
    CalibrationCountdown cd(3.0);
    CHECK(!cd.active());
    CHECK(!cd.update(1.0, true)); // not started: ignored
    cd.start();
    CHECK(cd.active());
    CHECK_NEAR(cd.remaining(), 3.0, 1e-9);
    CHECK(!cd.update(1.0, true));
    CHECK(!cd.update(1.0, true));
    CHECK_NEAR(cd.remaining(), 1.0, 1e-9);
    CHECK(!cd.update(0.5, false)); // lost a hand: restarts
    CHECK_NEAR(cd.remaining(), 3.0, 1e-9);
    CHECK(cd.active());
    int done = 0;
    for (int i = 0; i < 40; ++i) {
      done += cd.update(0.1, true);
    }
    CHECK(done == 1);
    CHECK(!cd.active());

    cd.start();
    cd.cancel();
    CHECK(!cd.active() && !cd.update(10.0, true));
  }

  // Help text lists every key app handles
  {
    std::string all;
    for (const auto &l : helpLines()) {
      all += l + "\n";
    }
    for (const char *k :
         {"w phase", "x pan", "c calibrate", "p pitch", "e effects", "s segment", "v bars",
          "r record", "l low-light", "t denoise", "d show", "q / Esc", "pinch", "upside down"}) {
      CHECK(all.find(k) != std::string::npos);
    }
  }

  // Overlays draw something and keep image geometry; tiny frames don't crash
  {
    cv::Mat base(480, 640, CV_8UC3, cv::Scalar(90, 90, 90)), v = base.clone();
    drawHelp(v);
    CHECK(v.size() == base.size() && changedPixels(base, v) > 5000);

    v = base.clone();
    CalibrationCountdown cd;
    drawCountdown(v, cd, true); // Inactive: draws nothing
    CHECK(changedPixels(base, v) == 0);
    cd.start();
    drawCountdown(v, cd, true);
    CHECK(changedPixels(base, v) > 100);
    cv::Mat w = base.clone();
    drawCountdown(w, cd, false);
    CHECK(changedPixels(base, w) > 100 && changedPixels(v, w) > 100);

    v = base.clone();
    drawFirstRunHint(v);
    CHECK(changedPixels(base, v) > 100);

    cv::Mat tiny(40, 60, CV_8UC3, cv::Scalar(0, 0, 0));
    drawHelp(tiny);
    drawCountdown(tiny, cd, true);
    drawFirstRunHint(tiny);
    CHECK(tiny.size() == cv::Size(60, 40));
  }

  // File picker: zenity first, then kdialog, "" when neither or cancelled
  {
    auto world = [](bool zenity, bool kdialog, const std::string &answer) {
      return [=](const std::string &cmd) -> std::string {
        if (cmd.rfind("command -v zenity", 0) == 0)
          return zenity ? "/usr/bin/zenity\n" : "";
        if (cmd.rfind("command -v kdialog", 0) == 0)
          return kdialog ? "/usr/bin/kdialog\n" : "";
        if (cmd.rfind("zenity", 0) == 0)
          return zenity ? answer : "";
        if (cmd.rfind("kdialog", 0) == 0)
          return kdialog ? answer : "";
        return "";
      };
    };
    CHECK(pickAudioFile(world(true, true, "/home/h/a b.wav\n")) == "/home/h/a b.wav");
    CHECK(pickAudioFile(world(false, true, "/m/x.flac\n")) == "/m/x.flac");
    CHECK(pickAudioFile(world(false, false, "/m/x.flac\n")).empty());
    CHECK(pickAudioFile(world(true, false, "")).empty()); // cancelled
  }

  TEST_MAIN_END();
}
