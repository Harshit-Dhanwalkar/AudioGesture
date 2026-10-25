#pragma once
// Brightens dark webcam frames for hand tracker (and for preview, to see what
// tracker sees). Two cheap steps, applied only when scene is dark:
//
//  1. Gamma lift. Output = 255 * (in/255)^g with g < 1 lifts shadows much more
//  than highlights. g is chosen so frame's mean luma lands near `target`:
//  (mean/255)^g = target  =>   g = ln(target) / ln(mean/255) and is clamped to
//  [0.4, 1] so a nearly black frame doesn't turn into pure noise
//  2. CLAHE on L channel of Lab.  Contrast-limited adaptive histogram
//  equalisation works on small tiles, so a hand lit by a monitor gets local
//  contrast even when rest of room is black; clip limit stops it blowing up sensor noise

#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <vector>

class LowLightEnhancer {
public:
  enum class Mode { Auto, On, Off };

  // Tunables (stages can be switched off individually; tools/hand_probe.cpp
  // sweeps them). CLAHE is off by default: in testing it amplified noise enough
  // to make detectors fail more often
  bool useGamma = true, useClahe = false;
  float maxBlurSigma =
      2.2f; // px; blur applied at pure black, fading to 0 at switch-off level

  void setMode(Mode m) { mode_ = m; }
  Mode mode() const { return mode_; }
  void cycleMode() {
    mode_ = (mode_ == Mode::Auto) ? Mode::On
            : (mode_ == Mode::On) ? Mode::Off
                                  : Mode::Auto;
  }
  bool active() const { return active_; }
  float meanLuma() const { return meanLuma_; } // 0..255, smoothed
  const char *modeName() const {
    return mode_ == Mode::Auto ? "auto" : mode_ == Mode::On ? "on" : "off";
  }

  // Returns either `bgr` itself (bright scene / disabled) or an internal
  // enhanced copy
  const cv::Mat &apply(const cv::Mat &bgr) {
    // Cheap brightness estimate on a 1/8-size gray image
    cv::resize(bgr, small_, cv::Size(), 0.125, 0.125, cv::INTER_AREA);
    cv::cvtColor(small_, gray_, cv::COLOR_BGR2GRAY);
    const float m = static_cast<float>(cv::mean(gray_)[0]);
    meanLuma_ = first_ ? m : 0.9f * meanLuma_ + 0.1f * m;
    first_ = false;

    switch (mode_) {
    case Mode::Off:
      active_ = false;
      break;
    case Mode::On:
      active_ = true;
      break;
    case Mode::Auto:
      if (!active_ && meanLuma_ < kOnBelow) {
        active_ = true;
      }
      if (active_ && meanLuma_ > kOffAbove) {
        active_ = false;
      }
      break;
    }

    if (!active_) {
      return bgr;
    }

    // 1. \gamma lift towards a mean luma of `kTarget`
    if (useGamma) {
      const float norm = std::max(meanLuma_, 4.f) / 255.f;
      const float g =
          std::clamp(std::log(kTarget) / std::log(norm), 0.4f, 1.0f);
      if (std::fabs(g - gamma_) > 0.02f || lut_.empty()) {
        gamma_ = g;
        lut_.create(1, 256, CV_8U);
        for (int i = 0; i < 256; ++i) {
          lut_.at<uchar>(i) =
              cv::saturate_cast<uchar>(255.0 * std::pow(i / 255.0, gamma_));
        }
      }
      cv::LUT(bgr, lut_, lifted_);
    } else {
      bgr.copyTo(lifted_);
    }

    // 1b. Lift amplifies sensor noise by same factor as signal and CNNs read
    //  that grain as texture. Hands are big smooth shapes, so average it away;
    //  darker scene, stronger blur (sigma ~2.2 px at black, ~0 near switch-off
    //  level)
    const float sigma = std::clamp(maxBlurSigma * (1.f - meanLuma_ / kOffAbove),
                                   0.f, maxBlurSigma);
    if (sigma > 0.4f) {
      cv::GaussianBlur(lifted_, lifted_, cv::Size(0, 0), sigma);
    }

    // 2. local contrast on lightness only (keeps colours stable)
    if (!useClahe) {
      out_ = lifted_;
      return out_;
    }

    if (!clahe_) {
      clahe_ = cv::createCLAHE(2.0, cv::Size(8, 8));
    }
    cv::cvtColor(lifted_, lab_, cv::COLOR_BGR2Lab);
    cv::split(lab_, planes_);
    clahe_->apply(planes_[0], planes_[0]);
    cv::merge(planes_, lab_);
    cv::cvtColor(lab_, out_, cv::COLOR_Lab2BGR);

    return out_;
  }

private:
  static constexpr float kOnBelow =
      85.f; // switch on below this mean luma (0..255)
  static constexpr float kOffAbove =
      105.f; // switch off above this (hysteresis)
  static constexpr float kTarget =
      0.42f; // desired mean luma after \gamma lift (0..1)

  Mode mode_ = Mode::Auto;
  bool active_ = false, first_ = true;
  float meanLuma_ = 128.f, gamma_ = 1.f;

  cv::Mat small_, gray_, lut_, lifted_, lab_, out_;
  std::vector<cv::Mat> planes_;
  cv::Ptr<cv::CLAHE> clahe_;
};
