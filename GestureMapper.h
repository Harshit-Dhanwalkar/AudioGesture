#pragma once
// Gesture -> DSP parameter mapping. All the mapping math lives here.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <opencv2/opencv.hpp>

#include "AudioEngine.h"
#include "GestureTracker.h"

class GestureMapper {
public:
    void update(const HandState& hs, double dt, AudioEngine& audio) {
        dt = std::max(dt, 1e-4);

        // ---- 1. Smooth raw hand coordinates (kills tracker jitter) ----
        // Exponential moving average with time constant tau: alpha = 1 - exp(-dt/tau).
        // Frame-rate independent, unlike a fixed alpha.
        const float a = 1.f - std::exp(-float(dt) / 0.08f);
        smooth(lx_, ly_, lInit_, hs.left, a);
        float prevRx = rx_;
        smooth(rx_, ry_, rInit_, hs.right, a);

        // ---- 2. Distance -> time stretch + grain size ----
        // Euclidean distance in *physical* units: scale x by the aspect ratio so a
        // 0.3 move horizontally equals a 0.3 move vertically on screen.
        //   d = sqrt( ((xr - xl) * aspect)^2 + (yr - yl)^2 )
        if (hs.left.valid && hs.right.valid) {
            const float dx = (rx_ - lx_) * hs.aspect, dy = ry_ - ly_;
            d_ = std::sqrt(dx * dx + dy * dy);
        }
        // Normalize usable range [0.10, 1.00] -> [0, 1].
        const float dn = std::clamp((d_ - 0.10f) / 0.90f, 0.f, 1.f);
        // Exponential map so equal hand motion = equal *musical* change:
        //   stretch = 2^(lerp(-2, +2, dn))  ->  0.25x (hands together) .. 4x (hands wide)
        // Hands apart = pulling the sound longer, like stretching rubber.
        p_.stretch = std::exp2(-2.f + 4.f * dn);
        // Longer stretch needs longer grains to stay smooth; short grains stay punchy.
        p_.grainMs = 40.f + 210.f * dn;

        // ---- 3. Vertical position -> pitch ----
        // y is 0 at the top of the frame. Map top -> +12 st, bottom -> -12 st (linear in
        // semitones, i.e. exponential in frequency, which matches how we hear pitch).
        if (hs.right.valid) p_.pitchSemis = 12.f - 24.f * std::clamp(ry_, 0.f, 1.f);

        // ---- 4. Horizontal velocity -> reverse / scrub ----
        // vx in frame-widths per second, low-passed. A fast sweep LEFT latches reverse,
        // a fast sweep RIGHT latches forward. The hysteresis (0.8) prevents flicker.
        if (hs.right.valid && rInitDone_) {
            const float vInst = (rx_ - prevRx) / float(dt);
            vx_ += (1.f - std::exp(-float(dt) / 0.05f)) * (vInst - vx_);
            if (vx_ < -0.8f) p_.direction = -1.f;
            else if (vx_ > 0.8f) p_.direction = 1.f;
        }
        rInitDone_ = hs.right.valid;
        // Engine smooths direction too, so the head decelerates through 0 (a short
        // "tape stop") instead of flipping instantly.

        audio.setParams(p_);
    }

    void draw(cv::Mat& img, const HandState& hs, double playhead) const {
        auto pt = [&](float x, float y) { return cv::Point(int(x * img.cols), int(y * img.rows)); };
        if (hs.left.valid)  cv::circle(img, pt(lx_, ly_), 12, {255, 128, 0}, 2);
        if (hs.right.valid) cv::circle(img, pt(rx_, ry_), 12, {0, 128, 255}, 2);
        if (hs.left.valid && hs.right.valid) cv::line(img, pt(lx_, ly_), pt(rx_, ry_), {255, 255, 255}, 1);
        char buf[160];
        std::snprintf(buf, sizeof buf, "stretch %.2fx  grain %.0fms  pitch %+.1fst  dir %+.0f",
                      p_.stretch, p_.grainMs, p_.pitchSemis, p_.direction);
        cv::putText(img, buf, {10, 24}, cv::FONT_HERSHEY_SIMPLEX, 0.55, {255, 255, 255}, 1, cv::LINE_AA);
        cv::rectangle(img, {10, img.rows - 20}, {img.cols - 10, img.rows - 10}, {80, 80, 80}, 1);
        cv::rectangle(img, {10, img.rows - 20}, {10 + int((img.cols - 20) * playhead), img.rows - 10}, {0, 255, 0}, cv::FILLED);
    }

private:
    static void smooth(float& x, float& y, bool& init, const Hand& h, float a) {
        if (!h.valid) return;
        if (!init) { x = h.x; y = h.y; init = true; return; }
        x += a * (h.x - x);
        y += a * (h.y - y);
    }

    EngineParams p_;
    float lx_ = 0.3f, ly_ = 0.5f, rx_ = 0.7f, ry_ = 0.5f, d_ = 0.5f, vx_ = 0.f;
    bool lInit_ = false, rInit_ = false, rInitDone_ = false;
};
