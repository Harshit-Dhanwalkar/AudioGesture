#pragma once
// Real-time granular audio engine
//
// Granular: Plain "resample playback" ties speed and pitch together
// With grains we get two independent controls:
//   * head speed  (how fast read position travels through file)  -> time
//   stretch / reverse / scrub
//   * grain pitch (how fast each short grain reads its samples)  -> pitch shift
// Each grain is a short Hann-windowed snippet, so there are no clicks at
// buffer boundaries or when parameters change

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <portaudio.h>
#include <string>
#include <vector>

#include "audiogesture/audio/Recorder.h"

struct EngineParams {
  float stretch = 1.0f;    // time-stretch factor: 2 = twice as long (half speed),
                           // 0.5 = double speed
  float grainMs = 120.0f;  // grain length in milliseconds
  float pitchSemis = 0.0f; // pitch shift in semitones
  float direction = 1.0f;  // +1 forward, -1 reverse, values between = scrub through zero
  float gain = 0.8f;
  float invert = 1.0f; // +1 normal, -1 phase-flipped (master)

  // "Rubber waveform" amplitude envelope. File is cut into segments of segSec
  // seconds; inside a segment gain slides linearly from envL (at its start,
  // under left hand) to envR (at its end, right hand)
  // Signed: +1 full, 0 silent, -1 full but phase-flipped (hand upside down)
  float envL = 1.0f, envR = 1.0f;
  float segSec = 4.0f;
};

class AudioEngine {
public:
  AudioEngine();
  ~AudioEngine();

  bool load(const std::string &path); // WAV/FLAC/OGG (and MP3 with libsndfile >= 1.1)
  bool start();
  void stop();

  // Headless use (unit tests, offline rendering): give the engine samples and a
  // sample rate without touching the disk or PortAudio, then pull audio with
  // render()
  void loadSamples(std::vector<float> mono, double fileSampleRate);
  void loadStereo(std::vector<float> left, std::vector<float> right, double fileSampleRate);
  void initOffline(double deviceSampleRate);

  // Fills `frames` interleaved stereo frames. Called from the PortAudio
  // callback; public so tests can drive it directly.
  void render(float *out, unsigned long frames);

  // Thread-safe setters: write "targets"; audio thread smooths toward them
  void setParams(const EngineParams &p);

  // Where the read head is inside the current segment (0 = left edge of the
  // grid, 1 = right edge). Updated once per audio buffer.
  float segmentPhase() const { return segPhase_.load(std::memory_order_relaxed); }

  double playheadNorm() const { return headNorm_.load(std::memory_order_relaxed); }
  bool isStereo() const { return !right_.empty(); }

  // Visualizer tap
  // Audio callback copies every output sample into a lock-free ring; UI thread
  // calls copyRecent() to read newest n samples (mono, oldest first, n <=
  // 16384).  Single writer, relaxed atomics per sample: a read can at worst see
  // a glitch at very oldest sample, never a torn value and audio thread never
  // waits on UI
  void copyRecent(float *dst, size_t n) const;
  double outputSampleRate() const { return devSR_; }

  // Record exactly what leaves the speakers (stereo, 32-bit float WAV)
  // audio thread only copies into a lock-free ring; a writer thread does disk I/O, so recording can
  // never glitch playback
  bool startRecording(const std::string &wavPath) { return rec_.start(wavPath, devSR_); }
  void stopRecording() { rec_.stop(); }
  bool recording() const { return rec_.active(); }
  double recordedSeconds() const { return rec_.seconds(); }
  size_t recordDropped() const { return rec_.dropped(); }
  double fileSampleRate() const { return fileSR_; }

private:
  static int paCallback(const void *, void *out, unsigned long frames,
                        const PaStreamCallbackTimeInfo *, PaStreamCallbackFlags, void *user);
  void spawnGrain();
  void setupSmoothing();

  struct Grain {
    bool active = false;
    double pos = 0.0;  // read position in source samples (fractional)
    double sign = 1.0; // +1 forward, -1 reverse (fixed at spawn)
    int age = 0;
    int len = 1;
  };
  static constexpr int kMaxGrains = 16;

  // Source audio, float
  std::vector<float> left_, right_;
  double fileSR_ = 44100.0;

  // PortAudio
  PaStream *stream_ = nullptr;
  double devSR_ = 44100.0;
  bool paInit_ = false;

  // Targets written by UI thread
  std::atomic<float> tStretch_{1.f}, tGrain_{120.f}, tPitch_{0.f}, tDir_{1.f}, tGain_{0.8f},
      tInvert_{1.f}, tEnvL_{1.f}, tEnvR_{1.f}, tSeg_{4.f};
  std::atomic<double> headNorm_{0.0};
  std::atomic<float> segPhase_{0.f};

  Recorder rec_;

  // Output ring for visualizer (see copyRecent)
  static constexpr size_t kRing = 1 << 15; // 32768 samples, power of two
  std::unique_ptr<std::atomic<float>[]> ring_;
  std::atomic<uint64_t> ringPos_{0};

  // Audio-thread-only state
  float stretch_ = 1.f;
  float grainMs_ = 120.f;
  float pitch_ = 0.f;
  float dir_ = 1.f;
  float gain_ = 0.8f;
  float invert_ = 1.f;
  float envL_ = 1.f;
  float envR_ = 1.f;
  double head_ = 0.0;
  int countdown_ = 0;
  double aSlow_ = 0.0;
  float aFast_ = 0.0; // one-pole smoothing coefficients
  std::array<Grain, kMaxGrains> grains_;
};
