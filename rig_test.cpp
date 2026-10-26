// Offline test: drives GestureMapper with synthetic hands, renders the rig to PNGs,
// and renders the audio engine offline to check the envelope + speed.
#include <opencv2/opencv.hpp>
#include <sstream>
#include <array>
#include <atomic>
#include <memory>
#include <string>
#include <vector>
#include <portaudio.h>
#include <complex>
#define private public
#include "AudioEngine.h"
#undef private
#define private public
#include "GestureMapper.h"
#undef private
#include <cstdio>
#include <sndfile.h>

static Hand mk(float cx, float wristY, float tipY) {
  Hand h; h.valid = h.hasSkeleton = true; h.x = cx; h.y = 0.5f * (wristY + tipY);
  for (auto &p : h.lm) p = {cx, 0.5f * (wristY + tipY)};
  h.lm[0] = {cx, wristY}; h.lm[12] = {cx, tipY};
  return h;
}
int main() {
  // 8 s test tone sweep file
  { SF_INFO i{}; i.samplerate = 44100; i.channels = 1; i.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
    SNDFILE *f = sf_open("/tmp/test_tone.wav", SFM_WRITE, &i);
    for (int n = 0; n < 44100 * 8; ++n) { float t = n / 44100.f; float s = 0.8f * std::sin(6.28318f * (200 + 100 * t) * t) * (0.6f + 0.4f * std::sin(6.28318f * 2 * t)); sf_write_float(f, &s, 1);} sf_close(f); }
  AudioEngine a; if (!a.load("/tmp/test_tone.wav")) return 1;
  a.devSR_ = 44100; a.aSlow_ = 1 - std::exp(-1.0 / (0.040 * 44100)); a.aFast_ = 1 - std::exp(-1.0 / (0.015 * 44100));
  GestureMapper m; HandState hs; hs.aspect = 4.f / 3.f;
  struct Case { const char *name; Hand L, R; };
  Case cases[] = {
    {"rect_close", mk(0.38f, 0.85f, 0.40f), mk(0.62f, 0.85f, 0.40f)},
    {"rect_apart", mk(0.15f, 0.85f, 0.40f), mk(0.85f, 0.85f, 0.40f)},
    {"trapezium",  mk(0.2f, 0.8f, 0.45f), mk(0.8f, 0.7f, 0.5f)},
    {"triangle",   mk(0.2f, 0.8f, 0.8f),  mk(0.8f, 0.8f, 0.4f)},
    {"inverted",   mk(0.2f, 0.4f, 0.75f), mk(0.8f, 0.4f, 0.75f)},
    {"twist",      mk(0.2f, 0.8f, 0.45f), mk(0.8f, 0.4f, 0.75f)},
  };
  for (auto &c : cases) {
    hs.left = c.L; hs.right = c.R;
    for (int i = 0; i < 60; ++i) m.update(hs, 1 / 30.0, a);
    a.head_ = 0.4 * 44100 * 4; // somewhere in segment 0..: 1.6s
    cv::Mat img(480, 640, CV_8UC3, cv::Scalar(40, 40, 40));
    a.headNorm_.store(a.head_ / a.mono_.size());
    m.draw(img, hs, a);
    cv::imwrite(std::string("/tmp/rig_") + c.name + ".png", img);
    // engine envelope: render 1s, report RMS in first/last quarter relative to input
    float tgt = 0; (void)tgt;
    std::printf("%-11s stretch=%.2f envL=%+.2f envR=%+.2f dir=%+.0f\n", c.name, m.p_.stretch, m.p_.envL, m.p_.envR, m.p_.direction);
  }
  // Audio check: envL=0 -> envR=1 over a 4 s segment, speed 1x (stretch 1). Measure RMS in 4 quarters of segment 0.
  EngineParams p; p.stretch = 1; p.grainMs = 90; p.envL = 0; p.envR = 1; p.segSec = 4; p.gain = 0.8f; p.direction = 1;
  AudioEngine b; b.load("/tmp/test_tone.wav"); b.devSR_ = 44100; b.aSlow_ = a.aSlow_; b.aFast_ = a.aFast_;
  b.setParams(p); b.stretch_ = 1; b.grainMs_ = 90; b.envL_ = 0; b.envR_ = 1; b.gain_ = 0.8f; b.dir_ = 1; b.invert_ = 1; b.head_ = 0;
  std::vector<float> out(2 * 256);
  double rms[4] = {0, 0, 0, 0}; long cnt[4] = {0, 0, 0, 0}; long total = 0;
  while (total < 44100 * 4) { b.render(out.data(), 256); for (int i = 0; i < 256; ++i) { int q = std::min(3, int((total + i) / 44100)); rms[q] += out[2 * i] * out[2 * i]; cnt[q]++; } total += 256; }
  std::printf("ramp 0->1 RMS per second: %.3f %.3f %.3f %.3f\n", std::sqrt(rms[0] / cnt[0]), std::sqrt(rms[1] / cnt[1]), std::sqrt(rms[2] / cnt[2]), std::sqrt(rms[3] / cnt[3]));
  // inverted: envL=envR=-1 -> output should be anti-phase vs +1
  auto render1 = [&](float e) { AudioEngine c; c.load("/tmp/test_tone.wav"); c.devSR_ = 44100; c.aSlow_ = a.aSlow_; c.aFast_ = a.aFast_;
    EngineParams q = p; q.envL = q.envR = e; c.setParams(q); c.stretch_ = 1; c.grainMs_ = 90; c.envL_ = c.envR_ = e; c.gain_ = 0.8f; c.dir_ = 1; c.invert_ = 1; c.head_ = 44100;
    std::vector<float> o(2 * 4096); c.render(o.data(), 4096); return o; };
  auto pos = render1(1), neg = render1(-1); double dot = 0, n1 = 0; for (int i = 0; i < 4096; ++i) { dot += pos[2*i] * neg[2*i]; n1 += pos[2*i] * pos[2*i]; }
  std::printf("invert correlation (expect -1): %.3f\n", dot / n1);
  // speed: head advance per second at hands apart vs close
  hs.left = cases[0].L; hs.right = cases[0].R; for (int i = 0; i < 90; ++i) m.update(hs, 1/30., a); float s1 = m.p_.stretch;
  hs.left = cases[1].L; hs.right = cases[1].R; for (int i = 0; i < 90; ++i) m.update(hs, 1/30., a); float s2 = m.p_.stretch;
  std::printf("stretch close=%.2f apart=%.2f (speed %.2fx -> %.2fx)\n", s1, s2, 1 / s1, 1 / s2);
}
