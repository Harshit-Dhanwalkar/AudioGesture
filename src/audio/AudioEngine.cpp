#include "audiogesture/audio/AudioEngine.h"

#include <portaudio.h>
#include <sndfile.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace {
constexpr double kTwoPi = 6.283185307179586;
inline float clampf(float v, float lo, float hi) {
  return std::min(hi, std::max(lo, v));
} // namespace
} // namespace

AudioEngine::AudioEngine() : ring_(new std::atomic<float>[kRing]) {
  for (size_t i = 0; i < kRing; ++i)
    ring_[i].store(0.f, std::memory_order_relaxed);
}

void AudioEngine::copyRecent(float *dst, size_t n) const {
  n = std::min(n, kRing);
  const uint64_t end = ringPos_.load(std::memory_order_acquire);
  for (size_t i = 0; i < n; ++i)
    dst[i] = ring_[(end - n + i) & (kRing - 1)].load(std::memory_order_relaxed);
}
AudioEngine::~AudioEngine() {
  stop();
  if (paInit_)
    Pa_Terminate();
}

bool AudioEngine::load(const std::string &path) {
  SF_INFO info{};
  SNDFILE *f = sf_open(path.c_str(), SFM_READ, &info);
  if (!f) {
    std::fprintf(stderr, "sf_open failed: %s\n", sf_strerror(nullptr));
    return false;
  }

  std::vector<float> inter(static_cast<size_t>(info.frames) * info.channels);
  sf_count_t got = sf_readf_float(f, inter.data(), info.frames);
  sf_close(f);

  // Keep stereo as stereo (first two channels); anything else is downmixed to mono
  // A mono file is stored once and plays on both ears
  const int ch = info.channels;
  left_.assign(static_cast<size_t>(got), 0.f);
  right_.clear();
  if (ch == 2) {
    right_.assign(static_cast<size_t>(got), 0.f);
    for (sf_count_t i = 0; i < got; ++i) {
      left_[i] = inter[i * 2];
      right_[i] = inter[i * 2 + 1];
    }
  } else {
    for (sf_count_t i = 0; i < got; ++i) {
      float s = 0.f;
      for (int c = 0; c < ch; ++c) {
        s += inter[i * ch + c];
      }
      left_[i] = s / static_cast<float>(ch);
    }
  }

  fileSR_ = info.samplerate;
  std::printf("Loaded %lld frames, %d ch, %d Hz\n", (long long)got, info.channels, info.samplerate);
  return got > 1;
}

bool AudioEngine::start() {
  if (Pa_Initialize() != paNoError) {
    return false;
  }

  paInit_ = true;

  const PaDeviceInfo *di = Pa_GetDeviceInfo(Pa_GetDefaultOutputDevice());
  if (!di) {
    return false;
  }
  devSR_ = di->defaultSampleRate;
  setupSmoothing();

  if (Pa_OpenDefaultStream(&stream_, 0, 2, paFloat32, devSR_, 256, &AudioEngine::paCallback,
                           this) != paNoError) {
    return false;
  }

  return Pa_StartStream(stream_) == paNoError;
}

// One-pole smoother: y += a * (target - y), a = 1 - \exp(-1 / (\tau * sr))
// Time constant tau sets how fast we follow jittery hand data: long = smooth
// but laggy, short = responsive. 40 ms / 15 ms is a good start
void AudioEngine::setupSmoothing() {
  aSlow_ = 1.0 - std::exp(-1.0 / (0.040 * devSR_));
  aFast_ = 1.0 - std::exp(-1.0 / (0.015 * devSR_));
}

void AudioEngine::initOffline(double deviceSampleRate) {
  devSR_ = deviceSampleRate;
  setupSmoothing();
}

void AudioEngine::loadSamples(std::vector<float> mono, double fileSampleRate) {
  left_ = std::move(mono);
  right_.clear();
  fileSR_ = fileSampleRate;
}

void AudioEngine::loadStereo(std::vector<float> left, std::vector<float> right,
                             double fileSampleRate) {
  const size_t n = std::min(left.size(), right.size());
  left.resize(n);
  right.resize(n);
  left_ = std::move(left);
  right_ = std::move(right);
  fileSR_ = fileSampleRate;
}

void AudioEngine::stop() {
  if (stream_) {
    Pa_StopStream(stream_);
    Pa_CloseStream(stream_);
    stream_ = nullptr;
  }
}

void AudioEngine::setParams(const EngineParams &p) {
  tStretch_.store(clampf(p.stretch, 0.05f, 16.f), std::memory_order_relaxed);
  tGrain_.store(clampf(p.grainMs, 10.f, 500.f), std::memory_order_relaxed);
  tPitch_.store(clampf(p.pitchSemis, -24.f, 24.f), std::memory_order_relaxed);
  tDir_.store(clampf(p.direction, -1.f, 1.f), std::memory_order_relaxed);
  tGain_.store(clampf(p.gain, 0.f, 1.f), std::memory_order_relaxed);
  tInvert_.store(clampf(p.invert, -1.f, 1.f), std::memory_order_relaxed);
  tEnvL_.store(clampf(p.envL, -1.f, 1.f), std::memory_order_relaxed);
  tEnvR_.store(clampf(p.envR, -1.f, 1.f), std::memory_order_relaxed);
  tSeg_.store(clampf(p.segSec, 0.25f, 30.f), std::memory_order_relaxed);
}

int AudioEngine::paCallback(const void *, void *out, unsigned long frames,
                            const PaStreamCallbackTimeInfo *, PaStreamCallbackFlags, void *user) {
  static_cast<AudioEngine *>(user)->render(static_cast<float *>(out), frames);
  return paContinue;
}

void AudioEngine::spawnGrain() {
  // Reuse a free slot, otherwise steal oldest grain
  Grain *slot = nullptr;
  for (auto &g : grains_) {
    if (!g.active) {
      slot = &g;
      break;
    }
  }
  if (!slot) {
    slot = &*std::max_element(grains_.begin(), grains_.end(),
                              [](const Grain &a, const Grain &b) { return a.age < b.age; });
  }

  const int len = std::max(128, static_cast<int>(grainMs_ * 0.001 * devSR_));
  slot->active = true;
  slot->pos = head_;                       // grain starts where read head currently is
  slot->sign = (dir_ >= 0.f) ? 1.0 : -1.0; // grains play backward too when reversed
  slot->age = 0;
  slot->len = len;

  // Hop = len / 2 -> 50% overlap
  // A periodic Hann window at 50% overlap sums to exactly 1.0, so output level
  // is constant (no amplitude ripple)
  countdown_ = len / 2;
}

void AudioEngine::render(float *out, unsigned long frames) {
  const size_t N = left_.size();
  const std::vector<float> &R = right_.empty() ? left_ : right_;
  if (N < 2) {
    std::fill(out, out + frames * 2, 0.f);
    return;
  }

  const float tS = tStretch_.load(std::memory_order_relaxed);
  const float tG = tGrain_.load(std::memory_order_relaxed);
  const float tP = tPitch_.load(std::memory_order_relaxed);
  const float tD = tDir_.load(std::memory_order_relaxed);
  const float tV = tGain_.load(std::memory_order_relaxed);
  const float tI = tInvert_.load(std::memory_order_relaxed);
  const float tEL = tEnvL_.load(std::memory_order_relaxed);
  const float tER = tEnvR_.load(std::memory_order_relaxed);
  const double segLen =
      std::max(256.0, static_cast<double>(tSeg_.load(std::memory_order_relaxed)) * fileSR_);

  const double srRatio = fileSR_ / devSR_; // source samples consumed per output sample at 1x
  uint64_t wpos = ringPos_.load(std::memory_order_relaxed);

  for (unsigned long i = 0; i < frames; ++i) {
    // parameter smoothing (per-sample, audio thread)
    stretch_ += static_cast<float>(aSlow_) * (tS - stretch_);
    grainMs_ += static_cast<float>(aSlow_) * (tG - grainMs_);
    pitch_ += static_cast<float>(aSlow_) * (tP - pitch_);
    dir_ +=
        static_cast<float>(aFast_) * (tD - dir_); // passes through 0 => tape-style slowdown/scrub
    gain_ += static_cast<float>(aFast_) * (tV - gain_);
    invert_ += static_cast<float>(aFast_) * (tI - invert_);
    envL_ += static_cast<float>(aFast_) * (tEL - envL_);
    envR_ += static_cast<float>(aFast_) * (tER - envR_);

    // Pitch in semitones -> playback-rate ratio: rate = 2^(semitones / 12)
    // +12 st = 2x rate (octave up), -12 st = 0.5x (octave down)
    const double pitchRate = std::exp2(pitch_ / 12.0);

    // Read-head speed (source samples per output sample)
    // Time-stretch factor S slows head by 1/S; direction flips/scrubs it
    head_ += static_cast<double>(dir_) * srRatio / std::max(0.05f, stretch_);
    if (head_ >= N) {
      head_ -= N;
    } else if (head_ < 0) {
      head_ += N;
    }

    if (--countdown_ <= 0) {
      spawnGrain();
    }

    // sum active grains
    float accL = 0.f;
    float accR = 0.f;
    for (auto &g : grains_) {
      if (!g.active) {
        continue;
      }
      // Hann window: w = 0.5 * (1 - \cos(2 * \pi * age/len))
      // zero at both edges => click-free
      const float w = 0.5f * (1.f - static_cast<float>(std::cos(kTwoPi * g.age / g.len)));

      // Linear-interpolated read with wrap-around at buffer edges
      const size_t i0 = static_cast<size_t>(g.pos) % N;
      const size_t i1 = (i0 + 1) % N;
      const float frac = static_cast<float>(g.pos - std::floor(g.pos));
      accL += w * (left_[i0] + frac * (left_[i1] - left_[i0]));
      accR += w * (R[i0] + frac * (R[i1] - R[i0]));

      g.pos += g.sign * pitchRate * srRatio; // grain pitch is independent of head speed
      if (g.pos >= N) {
        g.pos -= N;
      } else if (g.pos < 0) {
        g.pos += N;
      }
      if (++g.age >= g.len) {
        g.active = false;
      }
    }

    // Where is read head inside its segment? u in [0,1): left edge -> right edge
    // Gain is linear between edge values, so audio follows trapezoid drawn between hands
    const float u = static_cast<float>(std::fmod(head_, segLen) / segLen);
    const float env = envL_ + (envR_ - envL_) * u;

    const float sL = std::tanh(accL * gain_ * env) * invert_; // soft clip
    const float sR = std::tanh(accR * gain_ * env) * invert_;
    out[2 * i] = sL;
    out[2 * i + 1] = sR;
    ring_[wpos++ & (kRing - 1)].store(0.5f * (sL + sR),
                                      std::memory_order_relaxed); // visualizer tap (mono mix)
  }

  rec_.push(out, frames);
  ringPos_.store(wpos, std::memory_order_release);
  headNorm_.store(head_ / static_cast<double>(N), std::memory_order_relaxed);
  segPhase_.store(static_cast<float>(std::fmod(head_, segLen) / segLen), std::memory_order_relaxed);
}
