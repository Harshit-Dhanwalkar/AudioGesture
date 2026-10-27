// SpectrumGrid: bars stay inside the hand-spanned quad (trapezium / triangle /
// inverted), nothing is drawn without both hands.
#include <opencv2/imgproc.hpp>

#include "audiogesture/ui/SpectrumGrid.h"
#include "check.h"

namespace {
const int W = 640, H = 480;

RigGeometry geo(float lx, float ly, float rx, float ry, float eL, float eR) {
  RigGeometry g;
  g.valid = true;
  g.lx = lx; g.ly = ly; g.rx = rx; g.ry = ry; g.eL = eL; g.eR = eR;
  return g;
}

// count non-black pixels outside the polygon grown by `margin` px
int outside(const cv::Mat &img, const std::vector<cv::Point> &poly, int margin) {
  cv::Mat mask(img.size(), CV_8U, cv::Scalar(0));
  cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{poly}, cv::Scalar(255)); // bow-tie safe
  cv::dilate(mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, {2 * margin + 1, 2 * margin + 1}));
  cv::Mat gray;
  cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
  gray.setTo(0, mask);
  return cv::countNonZero(gray);
}

std::vector<cv::Point> quad(const RigGeometry &g) {
  auto c = [&](float x, float y, float e, float s) {
    return cv::Point(cvRound(x * W), cvRound((y - s * e * 0.5f) * H));
  };
  return {c(g.lx, g.ly, g.eL, -1), c(g.rx, g.ry, g.eR, -1), c(g.rx, g.ry, g.eR, 1), c(g.lx, g.ly, g.eL, 1)};
}
} // namespace

int main() {
  const std::vector<float> full(48, 1.f), none(48, 0.f);
  SpectrumGrid grid;

  // no hands: image untouched
  {
    cv::Mat img(H, W, CV_8UC3, cv::Scalar::all(0));
    RigGeometry g;
    grid.draw(img, g, full, none, 0.5f, 1.f);
    CHECK(cv::countNonZero(cv::Mat(img.reshape(1)) ) == 0);
  }

  const RigGeometry cases[] = {
      geo(0.2f, 0.5f, 0.8f, 0.5f, 0.4f, 0.4f),    // rectangle
      geo(0.2f, 0.6f, 0.8f, 0.4f, 0.5f, 0.2f),    // trapezium, tilted
      geo(0.2f, 0.7f, 0.8f, 0.5f, 0.0f, 0.5f),    // triangle (left fist)
      geo(0.2f, 0.5f, 0.8f, 0.5f, -0.4f, -0.4f),  // inverted
      geo(0.2f, 0.5f, 0.8f, 0.5f, 0.4f, -0.4f),   // twist
  };
  for (const auto &g : cases) {
    for (auto style : {0, 1}) {
      SpectrumGrid gr;
      if (style) gr.cycleStyle();
      cv::Mat img(H, W, CV_8UC3, cv::Scalar::all(0));
      gr.draw(img, g, full, full, 0.5f, 1.f);
      CHECK(cv::countNonZero(cv::Mat(img.reshape(1))) > 2000); // something was drawn
      // frame lines are 4 px wide + AA, handles are circles: allow 9 px slack
      CHECK(outside(img, quad(g), 9) == 0);
    }
  }

  // bars follow the local height: a tall edge fills more pixels than a short one
  {
    cv::Mat a(H, W, CV_8UC3, cv::Scalar::all(0)), b = a.clone();
    grid.draw(a, geo(0.2f, 0.5f, 0.8f, 0.5f, 0.4f, 0.4f), full, none, 0.5f, 1.f);
    grid.draw(b, geo(0.2f, 0.5f, 0.8f, 0.5f, 0.4f, 0.1f), full, none, 0.5f, 1.f);
    cv::Mat ga, gb;
    cv::cvtColor(a, ga, cv::COLOR_BGR2GRAY);
    cv::cvtColor(b, gb, cv::COLOR_BGR2GRAY);
    CHECK(cv::countNonZero(ga) > cv::countNonZero(gb));
  }

  TEST_MAIN_END();
}
