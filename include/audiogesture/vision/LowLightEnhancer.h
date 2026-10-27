#pragma once
// Brightens dark webcam frames for hand tracker (and for preview, to see what
// tracker sees). Two cheap steps, applied only when scene is dark:
//
//  1. Gamma lift. Output = 255 * (in/255)^g with g < 1 lifts shadows much more
//  than highlights. g is chosen so frame's mean luma lands near `target`:
//  (mean/255)^g = target  =>   g = ln(target) / ln(mean/255) and is clamped to
//  [0.4, 1] so a nearly black frame doesn't turn into pure noise
//  0. Motion-adaptive temporal denoise (before the lift, dark scenes only):
//  sensor noise is independent from frame to frame, the scene is not, so a
//  recursive average acc += a*(cur - acc) cuts noise by ~sqrt(1/a) where
//  nothing moves. Per pixel, a grows with |cur - acc| (a = d/(d+T), floor aMin)
//  so a moving hand is taken from the new frame almost as-is: no ghost trails.
//  2. CLAHE on L channel of Lab.  Contrast-limited adaptive histogram
//  equalisation works on small tiles, so a hand lit by a monitor gets local
//  contrast even when rest of room is black; clip limit stops it blowing up sensor noise

#include <algorithm>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

class LowLightEnhancer {
public:
  enum class Mode { Auto, On, Off };

  // Tunables (stages can be switched off individually; tools/hand_probe.cpp
  // sweeps them). CLAHE is off by default: in testing it amplified noise enough
  // to make detectors fail more often
  bool useGamma = true, useClahe = false;
  bool useTemporal = true;  // motion-adaptive frame averaging (see above)
  float temporalMin = 0.50f; // smallest blend weight of the new frame (~2-frame average;
                             // 0.3 smeared hands and hurt palm detection in tests)
  float temporalT = 14.f;    // 8-bit difference at which a = 0.5: above ~noise, below hand motion
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
      accValid_ = false; // history is stale once we leave dark mode
      return bgr;
    }

    // 0. temporal denoise on the raw 8-bit frame
    const cv::Mat *src = &bgr;
    if (useTemporal) {
      temporalDenoise(bgr);
      src = &dn_;
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
      cv::LUT(*src, lut_, lifted_);
    } else {
      src->copyTo(lifted_);
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
  void temporalDenoise(const cv::Mat &bgr) {
    bgr.convertTo(curF_, CV_32FC3);
    if (!accValid_ || acc_.size() != curF_.size()) {
      curF_.copyTo(acc_);
      accValid_ = true;
    } else {
      // per-pixel change, averaged over channels and blurred a little so a
      // single noisy pixel cannot open the gate on its own
      cv::absdiff(curF_, acc_, diff_);
      cv::transform(diff_, diff1_, cv::Matx13f(1.f / 3, 1.f / 3, 1.f / 3));
      cv::GaussianBlur(diff1_, diff1_, cv::Size(5, 5), 0);
      cv::add(diff1_, cv::Scalar(temporalT), den_);
      cv::divide(diff1_, den_, alpha_);
      cv::max(alpha_, temporalMin, alpha_);
      cv::merge(std::vector<cv::Mat>{alpha_, alpha_, alpha_}, alpha3_);
      cv::subtract(curF_, acc_, diff_);
      cv::multiply(diff_, alpha3_, diff_);
      cv::add(acc_, diff_, acc_);
    }
    acc_.convertTo(dn_, CV_8UC3);
  }

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
  cv::Mat curF_, acc_, diff_, diff1_, den_, alpha_, alpha3_, dn_;
  bool accValid_ = false;
  std::vector<cv::Mat> planes_;
  cv::Ptr<cv::CLAHE> clahe_;
};
