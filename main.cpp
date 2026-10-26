#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>

#include "AudioEngine.h"
#include "GestureMapper.h"
#include "GestureTracker.h"
#include "HandTracker.h"
#include "LowLightEnhancer.h"
#include "Visualizer.h"

namespace {

struct Args {
  std::string audio;
  std::string source =
      "0"; // camera index, or a video file (handy for testing without a webcam)
  std::string modelsDir;
  bool skin = false; // force old skin-colour tracker
};

void usage(const char *exe) {
  std::fprintf(stderr,
               "usage: %s <audio.wav|flac|ogg|mp3> [camera_index | video_file] "
               "[--models DIR] [--skin]\n"
               "  keys: l low-light boost (auto/on/off)   v visualizer style   "
               "d show tracking ROI   c calibrate pose   p pitch on/off   "
               "s segment length   q/Esc quit\n",
               exe);
}

bool isNumber(const std::string &s) {
  return !s.empty() && std::all_of(s.begin(), s.end(), [](unsigned char c) {
    return std::isdigit(c);
  });
}

bool parseArgs(int argc, char **argv, Args &a) {
  int positional = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    if (s == "--skin") {
      a.skin = true;
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
                 : std::make_unique<CameraHandSource>(
                       std::atoi(args.source.c_str()), cfg);
    if (!cam->isOpen()) {
      std::fprintf(stderr, "cannot open video source '%s'\n",
                   args.source.c_str());
      return 1;
    }

    if (cam->modelsOk()) {
      std::printf("hand tracker: MediaPipe landmarks via OpenCV %s (%s)\n",
                  CV_VERSION, cam->tracker().landmarkModelUsed().c_str());
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

    auto skin =
        std::make_unique<SkinHandTracker>(std::atoi(args.source.c_str()));
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

  GestureMapper mapper;
  Visualizer viz;
  std::vector<float> tap(viz.windowSize());
  HandState hands;
  bool debugRoi = false;
  double fps = 0;
  auto last = std::chrono::steady_clock::now();

  constexpr int kSpecH = 150;
  constexpr int kWaveH = 44;

  while (src->poll(hands)) {
    const auto now = std::chrono::steady_clock::now();
    const double dt = std::chrono::duration<double>(now - last).count();
    last = now;
    fps = (fps == 0) ? 1.0 / std::max(dt, 1e-3)
                     : 0.9 * fps + 0.1 / std::max(dt, 1e-3);

    mapper.update(hands, dt, audio);

    // camera view + skeleton + parameter HUD
    cv::Mat view = src->frame().clone();
    if (dnn)
      drawHands(view, hands, debugRoi);
    mapper.draw(view, hands, audio);
    char status[160];
    if (dnn) {
      const LowLightEnhancer &e = dnn->enhancer();
      std::snprintf(status, sizeof status,
                    "%.0f fps | light %.0f/255, boost %s (%s) | palm %.0f ms, "
                    "landmarks %.0f ms",
                    fps, e.meanLuma(), e.active() ? "ON" : "off", e.modeName(),
                    dnn->tracker().palmMs(), dnn->tracker().landmarkMs());
    } else {
      std::snprintf(status, sizeof status,
                    "%.0f fps | skin-colour tracker (run `make models` for "
                    "skeleton tracker)",
                    fps);
    }
    cv::putText(view, status, {10, 46}, cv::FONT_HERSHEY_SIMPLEX, 0.45,
                {0, 0, 0}, 3, cv::LINE_AA);
    cv::putText(view, status, {10, 46}, cv::FONT_HERSHEY_SIMPLEX, 0.45,
                {200, 255, 200}, 1, cv::LINE_AA);

    // audio panels underneath: spectrum of what is actually being played + file
    // overview
    audio.copyRecent(tap.data(), tap.size());
    viz.update(tap.data(), audio.outputSampleRate(), dt);
    cv::Mat canvas(view.rows + kSpecH + kWaveH, view.cols, CV_8UC3,
                   cv::Scalar(0, 0, 0));
    view.copyTo(canvas(cv::Rect(0, 0, view.cols, view.rows)));
    viz.drawSpectrum(canvas, cv::Rect(0, view.rows, view.cols, kSpecH));
    viz.drawWaveform(canvas, cv::Rect(0, view.rows + kSpecH, view.cols, kWaveH),
                     audio.samples(), audio.playheadNorm());

    cv::imshow("gesture_grain", canvas);
    int k = cv::waitKey(1);
    if (k >= 0) {
      k &= 0xFF;
    }
    if (k == 27 || k == 'q') {
      break;
    }
    if (k == 'v') {
      viz.cycleStyle();
    }
    if (k == 'd') {
      debugRoi = !debugRoi;
    }
    if (k == 'c') {
      mapper.calibrate(hands); // current pose = 1x speed, full amplitude
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
  }

  audio.stop();

  return 0;
}
