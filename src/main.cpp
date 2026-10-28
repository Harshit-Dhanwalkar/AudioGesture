#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/version.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

#include "audiogesture/audio/AudioEngine.h"
#include "audiogesture/audio/SpectrumAnalyzer.h"
#include "audiogesture/control/GestureMapper.h"
#include "audiogesture/control/Settings.h"
#include "audiogesture/ui/SpectrumGrid.h"
#include "audiogesture/vision/GestureTracker.h"
#include "audiogesture/vision/HandTracker.h"
#include "audiogesture/vision/LowLightEnhancer.h"

namespace {

struct Args {
  std::string audio;
  std::string source = "0"; // camera index, or a video file (handy for testing without a webcam)
  std::string modelsDir;
  bool skin = false;    // force old skin-colour tracker
  std::string record;   // start recording to this WAV immediately (also: key r)
  std::string snapshot; // headless smoke test: save view after ~40 frames and exit
};

void usage(const char *exe) {
  std::fprintf(
      stderr,
      "usage: %s <audio.wav|flac|ogg|mp3> [camera_index | video_file] "
      "[--models DIR] [--skin] [--snapshot out.png] [--record out.wav]\n"
      "  keys: l low-light boost (auto/on/off)   "
      "v bars/mirrored   d show tracking ROI   c calibrate pose\n"
      "  p pitch on/off   s segment length   t temporal denoise   r record WAV   q/Esc quit\n",
      exe);
}

bool isNumber(const std::string &s) {
  return !s.empty() &&
         std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c); });
}

bool parseArgs(int argc, char **argv, Args &a) {
  int positional = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    if (s == "--skin") {
      a.skin = true;
    } else if (s == "--record" && i + 1 < argc) {
      a.record = argv[++i];
    } else if (s == "--snapshot" && i + 1 < argc) {
      a.snapshot = argv[++i];
    } else if (s == "--models" && i + 1 < argc) {
      a.modelsDir = argv[++i];
    } else if (s == "-h" || s == "--help") {
      return false;
    } else if (positional == 0) {
      a.audio = s;
      ++positional;
    } else if (positional == 1) {
      a.source = s;
      ++positional;
    } else {
      return false;
    }
  }

  return !a.audio.empty();
}

void outlined(cv::Mat &img, const std::string &text, cv::Point at, double scale,
              const cv::Scalar &col) {
  cv::putText(img, text, at, cv::FONT_HERSHEY_SIMPLEX, scale, {0, 0, 0}, 3, cv::LINE_AA);
  cv::putText(img, text, at, cv::FONT_HERSHEY_SIMPLEX, scale, col, 1, cv::LINE_AA);
}

void drawHud(cv::Mat &view, const GestureMapper &mapper, const HandState &hands,
             const CameraHandSource *dnn, double fps) {
  outlined(view, mapper.status(), {10, 24}, 0.5, {255, 255, 255});
  char status[200];
  if (dnn) {
    const LowLightEnhancer &e = dnn->enhancer();
    std::snprintf(status, sizeof status,
                  "%.0f fps | light %.0f/255, boost %s (%s)%s | palm %.0f ms, landmarks %.0f ms%s",
                  fps, e.meanLuma(), e.active() ? "ON" : "off", e.modeName(),
                  e.active() && e.useTemporal ? " +denoise" : "", dnn->tracker().palmMs(),
                  dnn->tracker().landmarkMs(), dnn->tracker().scanMs() > 0 ? ", scan" : "");
  } else {
    std::snprintf(status, sizeof status,
                  "%.0f fps | skin-colour tracker (run `make models` for skeleton tracker)", fps);
  }

  outlined(view, status, {10, 46}, 0.45, {200, 255, 200});
  if (!(hands.left.valid && hands.right.valid)) {
    outlined(view, "show both hands: spectrum is stretched between them", {10, view.rows - 12}, 0.5,
             {255, 255, 255});
  }
}

// recordings/audiogesture-YYYYmmdd-HHMMSS.wav (directory is created)
std::string newRecordingPath() {
  std::filesystem::create_directories("recordings");
  char name[64];
  const std::time_t t = std::time(nullptr);
  std::strftime(name, sizeof name, "audiogesture-%Y%m%d-%H%M%S.wav", std::localtime(&t));

  return (std::filesystem::path("recordings") / name).string();
}

void drawRecBadge(cv::Mat &view, double seconds, size_t dropped) {
  const int m = int(seconds) / 60, s = int(seconds) % 60;
  char t[48];
  std::snprintf(t, sizeof t, "REC %d:%02d%s", m, s, dropped ? " (dropped!)" : "");
  const int x = view.cols - 130;
  cv::circle(view, {x, 20}, 7, {0, 0, 0}, cv::FILLED, cv::LINE_AA);
  cv::circle(view, {x, 20}, 5, {40, 40, 255}, cv::FILLED, cv::LINE_AA);
  outlined(view, t, {x + 14, 25}, 0.5, {255, 255, 255});
}
} // namespace

int main(int argc, char **argv) {
  Args args;
  if (!parseArgs(argc, argv, args)) {
    usage(argv[0]);
    return 1;
  }

  AudioEngine audio;
  if (!audio.load(args.audio))
    return 1;

  // Hand source: landmark tracker if models are there, else skin-colour
  // fallback
  std::unique_ptr<IHandSource> src;
  CameraHandSource *dnn = nullptr; // non-null when landmark tracker is in use
  const bool fromFile = !isNumber(args.source);
  if (!args.skin) {
    const std::string dir = findModelsDir(args.modelsDir);
    HandTrackerConfig cfg;
    cfg.palmModel = dir + "/" + kPalmModelFile;
    cfg.landmarkModel = dir + "/" + kHandModelFile;
    std::unique_ptr<CameraHandSource> cam =
        fromFile ? std::make_unique<CameraHandSource>(args.source, cfg)
                 : std::make_unique<CameraHandSource>(std::atoi(args.source.c_str()), cfg);
    if (!cam->isOpen()) {
      std::fprintf(stderr, "cannot open video source '%s'\n", args.source.c_str());
      return 1;
    }

    if (cam->modelsOk()) {
      std::printf("hand tracker: MediaPipe landmarks via OpenCV %s (%s)\n", CV_VERSION,
                  cam->tracker().landmarkModelUsed().c_str());
      dnn = cam.get();
      src = std::move(cam);
    } else {
      std::fprintf(stderr,
                   "hand tracker unavailable: %s\n"
                   "  -> run `make models` (or models/get_models.sh); falling "
                   "back to skin-colour tracker.\n",
                   cam->error().c_str());
    } // `cam` is destroyed here, releasing camera before fallback opens it
  }
  if (!src) {
    if (fromFile) {
      std::fprintf(stderr, "skin tracker only supports camera indices\n");
      return 1;
    }

    auto skin = std::make_unique<SkinHandTracker>(std::atoi(args.source.c_str()));
    if (!skin->isOpen()) {
      std::fprintf(stderr, "cannot open camera %s\n", args.source.c_str());
      return 1;
    }

    src = std::move(skin);
  }

  if (!audio.start()) {
    std::fprintf(stderr, "cannot start audio\n");
    return 1;
  }

  if (!args.record.empty() && !audio.startRecording(args.record)) {
    return 1;
  }

  GestureMapper mapper;
  SpectrumAnalyzer analyzer;
  SpectrumGrid grid;
  std::vector<float> tap(analyzer.windowSize());

  // Settings from the last run (calibration, segment length, toggles)
  const std::string settingsPath = Settings::defaultPath();
  Settings st;
  st.load(settingsPath);
  mapper.applySettings(st);
  if (st.gridStyle == 1) {
    grid.cycleStyle();
  }
  if (dnn) {
    dnn->enhancer().useTemporal = st.temporalDenoise;
    dnn->enhancer().setMode(st.boostMode == 1   ? LowLightEnhancer::Mode::On
                            : st.boostMode == 2 ? LowLightEnhancer::Mode::Off
                                                : LowLightEnhancer::Mode::Auto);
  }

  auto saveSettings = [&] {
    mapper.storeSettings(st);
    st.gridStyle = grid.style() == SpectrumGrid::Style::Mirrored ? 1 : 0;
    if (dnn) {
      st.temporalDenoise = dnn->enhancer().useTemporal;
      const auto md = dnn->enhancer().mode();
      st.boostMode = md == LowLightEnhancer::Mode::On    ? 1
                     : md == LowLightEnhancer::Mode::Off ? 2
                                                         : 0;
    }
    st.save(settingsPath);
  };

  HandState hands;
  bool debugRoi = false;
  double fps = 0;
  int frameNo = 0;
  auto last = std::chrono::steady_clock::now();

  while (src->poll(hands)) {
    const auto now = std::chrono::steady_clock::now();
    const double dt = std::chrono::duration<double>(now - last).count();
    last = now;
    fps = (fps == 0) ? 1.0 / std::max(dt, 1e-3) : 0.9 * fps + 0.1 / std::max(dt, 1e-3);

    audio.setParams(mapper.update(hands, dt));

    // Spectrum of what is actually leaving speakers
    audio.copyRecent(tap.data(), tap.size());
    analyzer.update(tap.data(), audio.outputSampleRate(), dt);

    // One view: camera + skeletons + spectrum drawn inside hand grid
    cv::Mat view = src->frame().clone();
    if (dnn) {
      drawHands(view, hands, debugRoi);
    }
    grid.draw(view, mapper.geometry(), analyzer.bars(), analyzer.peaks(), audio.segmentPhase(),
              mapper.params().direction);
    drawHud(view, mapper, hands, dnn, fps);
    if (audio.recording()) {
      drawRecBadge(view, audio.recordedSeconds(), audio.recordDropped());
    }

    if (!args.snapshot.empty()) {
      if (++frameNo >= 40) {
        cv::imwrite(args.snapshot, view);
        break;
      }
      continue; // no window needed
    }

    cv::imshow("audiogesture", view);
    int k = cv::waitKey(1);
    if (k >= 0) {
      k &= 0xFF;
    }
    if (k == 27 || k == 'q') {
      break;
    }
    if (k == 'v') {
      grid.cycleStyle();
    }
    if (k == 'd') {
      debugRoi = !debugRoi;
    }
    if (k == 'c') {
      mapper.calibrate(hands); // current pose = 1x speed, full amplitude
      saveSettings();          // ...and remember it for next time
    }
    if (k == 'r') {
      if (audio.recording()) {
        audio.stopRecording();
        std::printf("recording saved\n");
      } else {
        const std::string path = newRecordingPath();
        if (audio.startRecording(path)) {
          std::printf("recording to %s\n", path.c_str());
        }
      }
    }
    if (k == 'p') {
      mapper.togglePitch();
    }
    if (k == 's') {
      mapper.cycleSegment();
    }
    if (k == 'l' && dnn) {
      dnn->enhancer().cycleMode();
    }
    if (k == 't' && dnn) {
      dnn->enhancer().useTemporal = !dnn->enhancer().useTemporal;
    }
  }

  audio.stopRecording();
  audio.stop();
  if (args.snapshot.empty()) {
    saveSettings(); // headless smoke runs must not overwrite your settings
  }

  return 0;
}
