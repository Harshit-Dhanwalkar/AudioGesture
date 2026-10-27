// LowLightEnhancer: brightens dark frames, leaves bright ones alone, and the
// temporal denoise cuts noise on static areas without smearing moving ones.
#include <opencv2/imgproc.hpp>

#include "audiogesture/vision/LowLightEnhancer.h"
#include "check.h"

namespace {
cv::Mat noisy(const cv::Mat &clean, double sigma, cv::RNG &rng) {
  cv::Mat n(clean.size(), CV_16SC3), f16, out;
  rng.fill(n, cv::RNG::NORMAL, cv::Scalar::all(0), cv::Scalar::all(sigma));
  clean.convertTo(f16, CV_16SC3);
  f16 += n;
  f16.convertTo(out, CV_8UC3);
  return out;
}
double patchStd(const cv::Mat &img, cv::Rect r) {
  cv::Scalar m, s;
  cv::meanStdDev(img(r), m, s);
  return (s[0] + s[1] + s[2]) / 3.0;
}
} // namespace

int main() {
  cv::RNG rng(7);

  // brightens dark, leaves bright untouched (Auto mode)
  {
    LowLightEnhancer e;
    cv::Mat dark(240, 320, CV_8UC3, cv::Scalar::all(25));
    cv::Mat out;
    for (int i = 0; i < 30; ++i) out = e.apply(dark);
    CHECK(e.active());
    CHECK(cv::mean(out)[0] > 1.8 * 25);

    LowLightEnhancer f;
    cv::Mat bright(240, 320, CV_8UC3, cv::Scalar::all(160));
    for (int i = 0; i < 5; ++i) out = f.apply(bright);
    CHECK(!f.active());
    CHECK(out.data == bright.data); // returned as-is, no copy
  }

  // temporal denoise lowers noise on a static patch
  {
    cv::Mat clean(240, 320, CV_8UC3, cv::Scalar::all(22));
    LowLightEnhancer on, off;
    for (auto *e : {&on, &off}) { e->setMode(LowLightEnhancer::Mode::On); e->maxBlurSigma = 0.f; }
    on.useTemporal = true;
    off.useTemporal = false;
    cv::Mat a, b;
    for (int i = 0; i < 15; ++i) {
      const cv::Mat in = noisy(clean, 6.0, rng);
      a = on.apply(in).clone();
      b = off.apply(in).clone();
    }
    const cv::Rect patch(100, 80, 80, 60);
    CHECK(patchStd(a, patch) < 0.7 * patchStd(b, patch));
  }

  // a bright block that jumps is not smeared: the new position is mostly new frame
  {
    LowLightEnhancer on, off;
    for (auto *e : {&on, &off}) { e->setMode(LowLightEnhancer::Mode::On); e->maxBlurSigma = 0.f; }
    off.useTemporal = false;
    cv::Mat bg(240, 320, CV_8UC3, cv::Scalar::all(20));
    for (int i = 0; i < 10; ++i) { on.apply(noisy(bg, 4.0, rng)); off.apply(noisy(bg, 4.0, rng)); }
    cv::Mat moved = bg.clone();
    cv::rectangle(moved, cv::Rect(200, 100, 40, 40), cv::Scalar::all(120), cv::FILLED);
    const cv::Mat a = on.apply(moved).clone(), b = off.apply(moved).clone();
    const cv::Rect box(205, 105, 30, 30);
    CHECK(cv::mean(a(box))[0] > 0.6 * cv::mean(b(box))[0]);
  }

  TEST_MAIN_END();
}
