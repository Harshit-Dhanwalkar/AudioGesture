#include "audiogesture/ui/SpectrumGrid.h"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

namespace {

const cv::Scalar kLeftCol(255, 128, 0);  // BGR, matches the left-hand marker
const cv::Scalar kRightCol(0, 128, 255); // right-hand marker
const cv::Scalar kGrid(235, 235, 235);

cv::Scalar lerp(const cv::Scalar &a, const cv::Scalar &b, float t) {
  return a * (1.f - t) + b * t;
}

// cava-like vertical gradient: teal -> pink -> violet, by bar height
cv::Scalar heightColour(float t) {
  const cv::Scalar lo(255, 211, 41), mid(255, 92, 124), hi(163, 79, 255);
  t = std::clamp(t, 0.f, 1.f);
  return t < 0.55f ? lerp(lo, mid, t / 0.55f) : lerp(mid, hi, (t - 0.55f) / 0.45f);
}

} // namespace

void SpectrumGrid::draw(cv::Mat &img, const RigGeometry &g,
                        const std::vector<float> &bars, const std::vector<float> &peaks,
                        float playhead, float dirSign) const {
  if (!g.valid || bars.empty()) {
    return;
  }
  const int W = img.cols, H = img.rows;
  const cv::Point2f L(g.lx * W, g.ly * H), R(g.rx * W, g.ry * H);
  const float len = std::hypot(R.x - L.x, R.y - L.y);
  if (len < 12.f) {
    return;
  }

  // Centre line P(t) and half-height vector V(t), t = 0 (left hand) .. 1 (right).
  // Screen up is -y, so a positive edge height points up.
  auto P = [&](float t) { return L + (R - L) * t; };
  auto V = [&](float t) {
    const float e = g.eL + (g.eR - g.eL) * t;
    return cv::Point2f(0.f, -e * H * 0.5f);
  };
  // Point at fraction t along the grid and fraction f of its height
  // (f = 0 bottom edge, 1 top edge; for an inverted hand "bottom" is on top).
  auto G = [&](float t, float f) { return P(t) + V(t) * (2.f * f - 1.f); };
  auto pt = [](cv::Point2f p) { return cv::Point(cvRound(p.x), cvRound(p.y)); };

  const int B = int(bars.size());
  const bool mirrored = style_ == Style::Mirrored;

  // 1. translucent floor so the grid reads on any background
  {
    // fillPoly (not ...ConvexPoly): a twisted grid is a bow-tie, which is not convex
    const std::vector<std::vector<cv::Point>> quad = {
        {pt(G(0, 0)), pt(G(1, 0)), pt(G(1, 1)), pt(G(0, 1))}};
    cv::Mat over = img.clone();
    cv::fillPoly(over, quad, cv::Scalar(30, 24, 20), cv::LINE_AA);
    cv::addWeighted(over, 0.55, img, 0.45, 0, img);
  }

  // 2. bars, each warped to the local height of the grid
  const float slot = 1.f / float(B);
  const float gap = 0.12f * slot;
  for (int i = 0; i < B; ++i) {
    const float t0 = i * slot + gap * 0.5f, t1 = (i + 1) * slot - gap * 0.5f;
    const float v = std::clamp(bars[i], 0.f, 1.f);
    if (v <= 0.002f) {
      continue;
    }
    // Mirrored: grow both ways from the centre line; Bars: grow from bottom edge
    const float fa = mirrored ? 0.5f - 0.5f * v : 0.f;
    const float fb = mirrored ? 0.5f + 0.5f * v : v;
    const std::vector<std::vector<cv::Point>> poly = {
        {pt(G(t0, fa)), pt(G(t1, fa)), pt(G(t1, fb)), pt(G(t0, fb))}};
    cv::fillPoly(img, poly, heightColour(v), cv::LINE_AA);

    if (i < int(peaks.size())) {
      const float pk = std::clamp(peaks[i], 0.f, 1.f);
      if (pk > v + 0.02f) {
        const float fc = mirrored ? 0.5f + 0.5f * pk : pk;
        cv::line(img, pt(G(t0, fc)), pt(G(t1, fc)), {240, 240, 240}, 2, cv::LINE_AA);
        if (mirrored) {
          const float fd = 0.5f - 0.5f * pk;
          cv::line(img, pt(G(t0, fd)), pt(G(t1, fd)), {240, 240, 240}, 2, cv::LINE_AA);
        }
      }
    }
  }

  // 3. grid lines (they bend with the quad)
  auto thin = [&](cv::Point2f a, cv::Point2f b, int w, float alpha) {
    // faint lines via blending a colour towards the background
    cv::line(img, pt(a), pt(b), kGrid * alpha + cv::Scalar(20, 20, 20) * (1.f - alpha), w,
             cv::LINE_AA);
  };
  for (float f : {0.25f, 0.5f, 0.75f}) {
    thin(G(0, f), G(1, f), 1, f == 0.5f ? 0.55f : 0.3f);
  }
  for (int i = 8; i < B; i += 8) {
    thin(G(i * slot, 0), G(i * slot, 1), 1, 0.3f);
  }
  const std::vector<cv::Point> frame = {pt(G(0, 0)), pt(G(1, 0)), pt(G(1, 1)), pt(G(0, 1))};
  cv::polylines(img, frame, true, {0, 0, 0}, 4, cv::LINE_AA);
  cv::polylines(img, frame, true, kGrid, 1, cv::LINE_AA);

  // 4. hand edges with a marker at their "top" (fingertips); an inverted edge
  // shows the marker at the bottom
  auto handle = [&](float t, const cv::Scalar &col, float e, int textDx) {
    const cv::Point2f a = G(t, 0), b = G(t, 1);
    cv::line(img, pt(a), pt(b), {0, 0, 0}, 6, cv::LINE_AA);
    cv::line(img, pt(a), pt(b), col, 3, cv::LINE_AA);
    cv::circle(img, pt(P(t) + V(t)), 5, col, cv::FILLED, cv::LINE_AA);
    if (e < -0.02f) {
      cv::putText(img, "inv", {cvRound(P(t).x) + textDx, cvRound(P(t).y) + 4}, // inside the grid
                  cv::FONT_HERSHEY_SIMPLEX, 0.45, {255, 255, 255}, 1, cv::LINE_AA);
    }
  };
  handle(0.f, kLeftCol, g.eL, 8);
  handle(1.f, kRightCol, g.eR, -34);

  // 5. playhead + travel direction
  const float u = std::clamp(playhead, 0.f, 1.f);
  cv::line(img, pt(G(u, 0)), pt(G(u, 1)), {0, 0, 0}, 4, cv::LINE_AA);
  cv::line(img, pt(G(u, 0)), pt(G(u, 1)), {255, 255, 255}, 2, cv::LINE_AA);
  const cv::Point2f dir = (R - L) * (dirSign >= 0 ? 1.f : -1.f) / len;
  cv::arrowedLine(img, pt(P(u)), pt(P(u) + dir * 18.f), {255, 255, 255}, 2, cv::LINE_AA, 0,
                  0.5);
}
