#include "audiogesture/ui/Onboarding.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>

#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>

namespace {
void outlined(cv::Mat &img, const std::string &t, cv::Point at, double scale, const cv::Scalar &col,
              int thick = 1) {
  cv::putText(img, t, at, cv::FONT_HERSHEY_SIMPLEX, scale, {0, 0, 0}, thick + 2, cv::LINE_AA);
  cv::putText(img, t, at, cv::FONT_HERSHEY_SIMPLEX, scale, col, thick, cv::LINE_AA);
}
} // namespace

std::vector<std::string> helpLines() {
  return {
      "                          AUDIOGESTURE               ",
      "",
      "Both hands visible: spectrum fills shape between them",
      "  pull hands apart / together\tslow down / speed up",
      "  hand taller / shorter\tlouder / quieter at that edge",
      "  hand upside down\tinverts phase at that edge",
      "  sweep both hands sideways\treverse / forward",
      "  pinch thumb + index\tfreeze sound",
      "  turn left palm edge-on\thigh-pass (thin out bass)",
      "  turn right palm edge-on\tlow-pass (muffle highs)",
      "",
      "c calibrate (hold pose 3 s) p pitch from height   e effects on/off",
      "s segment length            v bars/mirrored       r record WAV",
      "w phase-aligned grains      l low-light boost     d show tracking boxes",
      "t denoise",
      "h or ? this help            q / Esc quit",
  };
}

void drawHelp(cv::Mat &view) {
  const auto lines = helpLines();
  const int lh = 22, pad = 16;
  auto width = [](const std::string &t) {
    int base = 0;
    return cv::getTextSize(t, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base).width;
  };

  // Lines with a tab are two columns: gesture, then what it does
  int col = 0;
  int w = 0;
  for (const auto &l : lines) {
    const auto tab = l.find('\t');
    if (tab != std::string::npos) {
      col = std::max(col, width(l.substr(0, tab)));
    }
  }

  col += 24;
  for (const auto &l : lines) {
    const auto tab = l.find('\t');
    w = std::max(w, tab == std::string::npos ? width(l) : col + width(l.substr(tab + 1)));
  }

  w += 2 * pad;
  const int h = int(lines.size()) * lh + 2 * pad;
  const cv::Rect box((view.cols - w) / 2, std::max(8, (view.rows - h) / 2), std::min(w, view.cols),
                     std::min(h, view.rows));
  cv::Mat roi = view(box & cv::Rect(0, 0, view.cols, view.rows));
  cv::Mat shade(roi.size(), roi.type(), cv::Scalar(20, 20, 20));
  cv::addWeighted(shade, 0.78, roi, 0.22, 0, roi);
  cv::rectangle(view, box, {200, 200, 200}, 1, cv::LINE_AA);

  int y = box.y + pad + 12;
  for (size_t i = 0; i < lines.size(); ++i, y += lh) {
    const std::string &l = lines[i];
    const cv::Scalar colr = i == 0 ? cv::Scalar(120, 230, 255) : cv::Scalar(240, 240, 240);
    const auto tab = l.find('\t');
    if (tab != std::string::npos) {
      outlined(view, "  " + l.substr(0, tab), {box.x + pad, y}, 0.5, colr);
      outlined(view, l.substr(tab + 1), {box.x + pad + col, y}, 0.5, {160, 235, 160});
    } else if (!l.empty()) {
      outlined(view, l, {box.x + pad, y}, 0.5, colr);
    }
  }
}

void drawCountdown(cv::Mat &view, const CalibrationCountdown &cd, bool bothHands) {
  if (!cd.active()) {
    return;
  }

  const cv::Point c(view.cols / 2, view.rows / 2);
  if (!bothHands) {
    outlined(view, "show both hands to calibrate", {c.x - 190, c.y}, 0.8, {80, 200, 255}, 2);
    return;
  }

  char t[64];
  std::snprintf(t, sizeof t, "%d", int(cd.remaining()) + 1);
  outlined(view, "hold your pose", {c.x - 110, c.y - 50}, 0.8, {255, 255, 255}, 2);
  outlined(view, t, {c.x - 20, c.y + 30}, 2.5, {120, 230, 255}, 4);
}

void drawFirstRunHint(cv::Mat &view) {
  outlined(view,
           "first run: hold both hands in a comfortable pose and press c to calibrate (h = help)",
           {10, view.rows - 12}, 0.5, {120, 230, 255});
}
