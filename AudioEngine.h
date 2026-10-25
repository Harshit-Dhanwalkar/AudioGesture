#pragma once
// Real-time granular audio engine.
//
// Why granular? Plain "resample playback" ties speed and pitch together.
// With grains we get two independent controls:
//   * head speed  (how fast the read position travels through the file)  -> time stretch / reverse / scrub
//   * grain pitch (how fast each short grain reads its samples)          -> pitch shift
// Each grain is a short Hann-windowed snippet, so there are no clicks at
// buffer boundaries or when parameters change.

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <portaudio.h>

struct EngineParams {
    float stretch    = 1.0f;   // time-stretch factor: 2 = twice as long (half speed), 0.5 = double speed
    float grainMs    = 120.0f; // grain length in milliseconds
    float pitchSemis = 0.0f;   // pitch shift in semitones
    float direction  = 1.0f;   // +1 forward, -1 reverse, values between = scrub through zero
    float gain       = 0.8f;
};

class AudioEngine {
public:
    AudioEngine();
    ~AudioEngine();

    bool load(const std::string& path);   // WAV/FLAC/OGG (and MP3 with libsndfile >= 1.1)
    bool start();
    void stop();

    // Thread-safe setters: write "targets"; the audio thread smooths toward them.
    void setParams(const EngineParams& p);

    double playheadNorm() const { return headNorm_.load(std::memory_order_relaxed); }
    const std::vector<float>& samples() const { return mono_; }

    // --- visualizer tap -------------------------------------------------------------------
    // The audio callback copies every output sample into a lock-free ring; the UI thread calls
    // copyRecent() to read the newest n samples (mono, oldest first, n <= 16384).  Single writer,
    // relaxed atomics per sample: a read can at worst see a glitch at the very oldest sample,
    // never a torn value, and the audio thread never waits on the UI.
    void   copyRecent(float* dst, size_t n) const;
    double outputSampleRate() const { return devSR_; }

private:
    static int paCallback(const void*, void* out, unsigned long frames,
                          const PaStreamCallbackTimeInfo*, PaStreamCallbackFlags, void* user);
    void render(float* out, unsigned long frames);
    void spawnGrain();

    struct Grain {
        bool   active = false;
        double pos    = 0.0;   // read position in source samples (fractional)
        double sign   = 1.0;   // +1 forward, -1 reverse (fixed at spawn)
        int    age    = 0;
        int    len    = 1;
    };
    static constexpr int kMaxGrains = 16;

    // Source audio (mono, float).
    std::vector<float> mono_;
    double fileSR_ = 44100.0;

    // PortAudio.
    PaStream* stream_ = nullptr;
    double devSR_ = 44100.0;
    bool paInit_ = false;

    // Targets written by the UI thread.
    std::atomic<float> tStretch_{1.f}, tGrain_{120.f}, tPitch_{0.f}, tDir_{1.f}, tGain_{0.8f};
    std::atomic<double> headNorm_{0.0};

    // Output ring for the visualizer (see copyRecent).
    static constexpr size_t kRing = 1 << 15;   // 32768 samples, power of two
    std::unique_ptr<std::atomic<float>[]> ring_;
    std::atomic<uint64_t> ringPos_{0};

    // Audio-thread-only state.
    float  stretch_ = 1.f, grainMs_ = 120.f, pitch_ = 0.f, dir_ = 1.f, gain_ = 0.8f;
    double head_ = 0.0;
    int    countdown_ = 0;
    double aSlow_ = 0.0, aFast_ = 0.0; // one-pole smoothing coefficients
    std::array<Grain, kMaxGrains> grains_;
};
