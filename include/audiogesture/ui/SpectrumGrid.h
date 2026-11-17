#pragma once
// Hand-held grid: audio spectrum drawn *inside* quadrilateral that two hands span, so you can see
// how your hands deform sound
//
//   * left edge  = left hand, right edge = right hand. Each edge is a vertical segment whose height
//     is that hand's vertical extent (wrist -> middle fingertip, see GestureMapper). Different
//     heights give a trapezium, a fist gives a triangle, an upside-down hand flips that edge
//   * grid lines are quad itself: frame, a 25/50/75 % height ladder and a vertical line every 8
//     bars, all interpolated between two edges, so they bend with shape
//   * bars are live spectrum of what is coming out of speakers (log frequency, low on the left,
//     high on right). Bar i stands on grid's bottom edge and reaches `bars[i] * local height` up,
//     so it is literally squashed by a short hand and hangs upside down under a flipped one
//   * a white playhead marks where read head is inside segment: that is position at which engine
//     currently samples gain ramp
//
// No random "wave" line is drawn: everything visible is either grid or
// measured audio.

#include <vector>

#include <opencv2/core.hpp>

// Where two hands put grid
// x/y are normalized (x in frame widths, y in frame heights); e* are signed edge heights in frame
// heights (+ = fingers up)
struct RigGeometry {
  bool valid = false;
  float lx = 0.3f, ly = 0.5f, rx = 0.7f, ry = 0.5f;
  float eL = 0.3f, eR = 0.3f;
};

class SpectrumGrid {
public:
  enum class Style { Bars, Mirrored };

  void cycleStyle() { style_ = (style_ == Style::Bars) ? Style::Mirrored : Style::Bars; }
  Style style() const { return style_; }

  // bars/peaks: SpectrumAnalyzer output (0..1). playhead: 0..1 position of read head inside current
  // segment
  // Return dirSign: +1 forward, -1 reverse
  void draw(cv::Mat &img, const RigGeometry &g, const std::vector<float> &bars,
            const std::vector<float> &peaks, float playhead, float dirSign,
            bool frozen = false) const;

private:
  Style style_ = Style::Bars;
};
