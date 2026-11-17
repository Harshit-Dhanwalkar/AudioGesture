// Recorder: what engine plays is what lands in WAV, bit for bit
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <thread>
#include <vector>

#include <sndfile.h>

#include "audiogesture/audio/AudioEngine.h"
#include "check.h"

int main() {
  const double sr = 44100.0;
  std::vector<float> s(size_t(4 * sr));
  for (size_t n = 0; n < s.size(); ++n) {
    s[n] = float(0.7 * std::sin(6.283185307 * 330.0 * n / sr));
  }
  AudioEngine e;
  e.loadSamples(s, sr);
  e.initOffline(sr);

  const auto path = (std::filesystem::temp_directory_path() / "audiogesture_rec_test.wav").string();
  CHECK(e.startRecording(path));
  CHECK(e.recording());
  CHECK(!e.startRecording(path)); // already recording

  // "Play" 1.5 s in 256-frame callbacks, remembering what was rendered
  std::vector<float> played, buf(2 * 256);
  for (int i = 0; i < int(1.5 * sr / 256); ++i) {
    e.render(buf.data(), 256);
    played.insert(played.end(), buf.begin(), buf.end());
    if (i % 40 == 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  }

  e.stopRecording();
  CHECK(!e.recording());
  CHECK(e.recordDropped() == 0);
  e.stopRecording(); // second stop is harmless

  SF_INFO info{};
  SNDFILE *f = sf_open(path.c_str(), SFM_READ, &info);
  CHECK(f != nullptr);
  if (f) {
    CHECK(info.channels == 2);
    CHECK(info.samplerate == 44100);
    CHECK((size_t)info.frames * 2 == played.size());
    std::vector<float> back(played.size());
    sf_readf_float(f, back.data(), info.frames);
    sf_close(f);
    size_t bad = 0;
    for (size_t i = 0; i < played.size(); ++i) {
      bad += back[i] != played[i];
    }
    CHECK(bad == 0); // 32-bit float WAV: exact
  }
  std::remove(path.c_str());

  // A path that cannot be opened fails cleanly
  CHECK(!e.startRecording("/nonexistent-dir/x.wav"));
  CHECK(!e.recording());

  TEST_MAIN_END();
}
