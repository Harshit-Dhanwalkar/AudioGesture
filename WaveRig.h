#pragma once
// "Rubber waveform" drawn between the two hands.
//
// The song is cut into segments of `segSec` seconds. The segment under the read
// head is drawn as a waveform whose left edge is pinned to the left hand and
// whose right edge is pinned to the right hand:
//
//   * pull the hands apart  -> the segment is drawn longer; the read head crosses
//     it more slowly (GestureMapper makes speed = 1/length, so what you see is
//     what you hear)
//   * each edge's height is that hand's vertical extent (wrist -> middle
//     fingertip), so a tall open hand = loud, a fist = silent. Different heights
//     give a trapezium, one closed fist a triangle
//   * hand upside down -> the edge height goes negative -> the waveform is
//     drawn flipped and the audio is phase-inverted
//
// Between the two edges, height and gain are interpolated linearly. The same
// two numbers (envL, envR) are sent to the audio engine, which applies the
// same interpolation at the read head's position inside the segment.

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/imgproc.hpp>

struct RigGeometry {
  bool valid = false;
  // Hand edge centres, normalized: x in frame widths, y in frame heights.
  float lx = 0.3f, ly = 0.5f, rx = 0.7f, ry = 0.5f;
  // Signed edge heights in frame heights (+ = fingers up). Already clamped to
  // the range the audio uses, so drawing == hearing.
  float eL = 0.3f, eR = 0.3f;
  float aspect = 4.f / 3.f;
};

class WaveRig {
public:
  // song: whole mono file. head01: read head 0..1 of the file.
  void draw(cv::Mat &img, const RigGeometry &g, const std::vector<float> &song,
            double fileSR, double segSec, double head01, float dirSign) {
    if (!g.valid || song.size() < 2) {
      return;
    }
    const int W = img.cols, H = img.rows;
    const cv::Point2f L(g.lx * W, g.ly * H), R(g.rx * W, g.ry * H);
    if (std::hypot(R.x - L.x, R.y - L.y) < 12.f) {
      return;
    }

    // Segment under the read head, and where the head is inside it
    const double segLen = std::max(256.0, segSec * fileSR);
    const double headPos = head01 * double(song.size());
    const long seg = long(std::floor(headPos / segLen));
    const float u = float((headPos - double(seg) * segLen) / segLen);
    rebuild(song, segLen, seg);

    // Edge vector: signed half-height in pixels (screen up = -y)
    auto edge = [&](float t) {
      const float e = g.eL + (g.eR - g.eL) * t;
      return cv::Point2f(0.f, -e * H * 0.5f);
    };

    const int C = int(cols_);
    std::vector<cv::Point> top(C), bot(C);
    for (int i = 0; i < C; ++i) {
      const float t = float(i) / float(C - 1);
      const cv::Point2f P = L + (R - L) * t, V = edge(t);
      top[i] = cv::Point(cvRound(P.x + V.x * mx_[i]), cvRound(P.y + V.y * mx_[i]));
      bot[i] = cv::Point(cvRound(P.x + V.x * mn_[i]), cvRound(P.y + V.y * mn_[i]));
    }
    std::vector<cv::Point> poly(top);
    poly.insert(poly.end(), bot.rbegin(), bot.rend());

    // Translucent body
    cv::Mat over = img.clone();
    cv::fillPoly(over, std::vector<std::vector<cv::Point>>{poly},
                 cv::Scalar(200, 240, 140), cv::LINE_AA);
    cv::addWeighted(over, 0.45, img, 0.55, 0, img);

    // Quad frame (shows the trapezium) + baseline
    const cv::Point2f vl = edge(0.f), vr = edge(1.f);
    const cv::Point q[4] = {cv::Point(L + vl), cv::Point(R + vr), cv::Point(R - vr),
                            cv::Point(L - vl)};
    for (int i = 0; i < 4; ++i) {
      cv::line(img, q[i], q[(i + 1) % 4], {255, 255, 255}, 1, cv::LINE_AA);
    }
    cv::line(img, cv::Point(L), cv::Point(R), {255, 255, 255}, 1, cv::LINE_AA);

    // Outline of the waveform
    cv::polylines(img, top, false, {0, 0, 0}, 3, cv::LINE_AA);
    cv::polylines(img, bot, false, {0, 0, 0}, 3, cv::LINE_AA);
    cv::polylines(img, top, false, {200, 240, 140}, 1, cv::LINE_AA);
    cv::polylines(img, bot, false, {200, 240, 140}, 1, cv::LINE_AA);

    // Hand edges (handles)
    auto handle = [&](cv::Point2f c, cv::Point2f v, const cv::Scalar &col, float e) {
      cv::line(img, cv::Point(c + v), cv::Point(c - v), {0, 0, 0}, 6, cv::LINE_AA);
      cv::line(img, cv::Point(c + v), cv::Point(c - v), col, 3, cv::LINE_AA);
      cv::circle(img, cv::Point(c + v), 5, col, cv::FILLED, cv::LINE_AA); // "top" marker
      if (e < -0.02f) {
        cv::putText(img, "inv", {int(c.x) - 12, int(c.y) + 4}, cv::FONT_HERSHEY_SIMPLEX,
                    0.45, {255, 255, 255}, 1, cv::LINE_AA);
      }
    };
    handle(L, vl, {255, 128, 0}, g.eL);
    handle(R, vr, {0, 128, 255}, g.eR);

    // Playhead inside the segment
    const cv::Point2f P = L + (R - L) * u, V = edge(u);
    cv::line(img, cv::Point(P + V), cv::Point(P - V), {0, 0, 0}, 4, cv::LINE_AA);
    cv::line(img, cv::Point(P + V), cv::Point(P - V), {255, 255, 255}, 2, cv::LINE_AA);
    // little arrow showing travel direction
    const cv::Point2f dir = (R - L) * (dirSign >= 0 ? 1.f : -1.f) / std::hypot(R.x - L.x, R.y - L.y);
    cv::arrowedLine(img, cv::Point(P), cv::Point(P + dir * 18.f), {255, 255, 255}, 2,
                    cv::LINE_AA, 0, 0.5);
  }

private:
  // Per-column min/max of segment `seg`, normalised by the file's peak
  void rebuild(const std::vector<float> &song, double segLen, long seg) {
    if (seg == seg_ && segLen == segLen_ && song.size() == n_ && mn_.size() == cols_) {
      return;
    }
    if (song.size() != n_) {
      peak_ = 1e-4f;
      const size_t stride = std::max<size_t>(1, song.size() / 200000);
      for (size_t i = 0; i < song.size(); i += stride) {
        peak_ = std::max(peak_, std::fabs(song[i]));
      }
    }
    seg_ = seg;
    segLen_ = segLen;
    n_ = song.size();
    mn_.assign(cols_, 0.f);
    mx_.assign(cols_, 0.f);
    const double start = double(seg) * segLen;
    for (size_t c = 0; c < cols_; ++c) {
      const double a = start + segLen * double(c) / double(cols_);
      const double b = start + segLen * double(c + 1) / double(cols_);
      const size_t i0 = size_t(std::max(0.0, a)), i1 = std::min(song.size(), size_t(std::max(0.0, b)));
      const size_t stride = std::max<size_t>(1, (i1 > i0 ? i1 - i0 : 1) / 64);
      float lo = 0.f, hi = 0.f;
      for (size_t i = i0; i < i1; i += stride) {
        lo = std::min(lo, song[i]);
        hi = std::max(hi, song[i]);
      }
      mn_[c] = std::clamp(lo / peak_, -1.f, 0.f);
      mx_[c] = std::clamp(hi / peak_, 0.f, 1.f);
    }
  }

  static constexpr size_t cols_ = 160;
  long seg_ = -1;
  double segLen_ = 0;
  size_t n_ = 0;
  float peak_ = 1e-4f;
  std::vector<float> mn_, mx_;
};
