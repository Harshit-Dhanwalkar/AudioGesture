// hand_probe - run landmark hand tracker on a still image or a video file,
// without a camera, audio or a window.  Useful to check that models load, to
// see what tracker does in bad light, and to measure speed.
//
//   hand_probe photo.jpg --out annotated.png
//   hand_probe photo.jpg --crop 0,320,548,525 --dark 0.15 --noise 8 --lowlight
//   --out dark.png hand_probe clip.mp4 --frames 120 --out last_frame.png
//
//   --models DIR     where the .onnx files live (default: search ./models etc.)
//   --crop x,y,w,h   use only this part of the image
//   --dark F         multiply brightness by F (0.1 = very dark) to simulate a
//   dim room
//   --noise S        add Gaussian sensor noise with std-dev S (0..255 scale)
//   after darkening
//   --lowlight       run the LowLightEnhancer (forced on) before tracking
//   --frames N       process N frames (an image is repeated, which exercises
//   ROI tracking)
//   --debug          draw the tracking ROI
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "HandTracker.h"

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s <image|video> [options]  (see top of "
                 "tools/hand_probe.cpp)\n",
                 argv[0]);
    return 1;
  }

  std::string in = argv[1], out = "hand_probe.png", models;
  cv::Rect crop;
  double dark = 1.0, noise = 0.0;
  bool lowlight = false, debug = false;
  std::string enhStages = "gamma,blur";
  double palmThr = 0.3, startThr = 0.5;
  int frames = 3;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> const char * {
      return (i + 1 < argc) ? argv[++i] : "";
    };

    if (a == "--out") {
      out = next();
    } else if (a == "--models") {
      models = next();
    } else if (a == "--crop") {
      std::sscanf(next(), "%d,%d,%d,%d", &crop.x, &crop.y, &crop.width,
                  &crop.height);
    } else if (a == "--dark") {
      dark = std::atof(next());
    } else if (a == "--noise") {
      noise = std::atof(next());
    } else if (a == "--frames") {
      frames = std::atoi(next());
    } else if (a == "--palm-thr") {
      palmThr = std::atof(next());
    } else if (a == "--start-thr") {
      startThr = std::atof(next());
    } else if (a == "--lowlight") {
      lowlight = true;
    } else if (a == "--enh") {
      lowlight = true;
      enhStages = next();
    } // e.g. gamma,blur  (subset of gamma,blur,clahe)
    else if (a == "--debug") {
      debug = true;
    }
  }

  cv::VideoCapture cap;
  cv::Mat still = cv::imread(in);
  if (still.empty()) {
    cap.open(in);
    if (!cap.isOpened()) {
      std::fprintf(stderr, "cannot read %s\n", in.c_str());
      return 1;
    }
  }

  const std::string dir = findModelsDir(models);
  HandTrackerConfig cfg;
  cfg.palmModel = dir + "/" + kPalmModelFile;
  cfg.landmarkModel = dir + "/" + kHandModelFile;
  cfg.palmScore = float(palmThr);
  cfg.presenceStart = float(startThr);
  LandmarkHandTracker tracker(cfg);
  if (!tracker.ok()) {
    std::fprintf(stderr, "tracker: %s\n", tracker.error().c_str());
    return 2;
  }
  std::printf("OpenCV %s, landmark model: %s\n", CV_VERSION,
              tracker.landmarkModelUsed().c_str());

  LowLightEnhancer enh;
  enh.setMode(lowlight ? LowLightEnhancer::Mode::On
                       : LowLightEnhancer::Mode::Off);
  enh.useGamma = enhStages.find("gamma") != std::string::npos;
  enh.useClahe = enhStages.find("clahe") != std::string::npos;
  enh.maxBlurSigma = enhStages.find("blur") != std::string::npos ? 2.2f : 0.f;
  cv::RNG rng(1234);

  cv::Mat view;
  HandState hs;
  int bothFrames = 0, anyFrames = 0, total = 0;
  double scoreSum = 0;
  int scoreN = 0;
  for (int f = 0; f < frames; ++f) {
    cv::Mat frame = still.empty() ? cv::Mat() : still.clone();
    if (still.empty() && (!cap.read(frame) || frame.empty())) {
      break;
    }
    if (crop.area() > 0) {
      frame = frame(crop & cv::Rect(0, 0, frame.cols, frame.rows)).clone();
    }
    if (dark != 1.0) {
      frame.convertTo(frame, -1, dark, 0);
    }
    if (noise > 0) {
      cv::Mat n(frame.size(), CV_16SC3);
      rng.fill(n, cv::RNG::NORMAL, cv::Scalar::all(0), cv::Scalar::all(noise));
      cv::Mat f16;
      frame.convertTo(f16, CV_16SC3);
      f16 += n;
      f16.convertTo(frame, CV_8UC3);
    }

    const cv::Mat &proc = enh.apply(frame);
    tracker.process(proc, 1.0 / 30.0, hs);
    std::printf(
        "frame %d: luma %.0f%s  palm %.1f ms  landmarks %.1f ms | L:%s R:%s", f,
        cv::mean(cv::Mat(frame))[0], lowlight ? "(enh)" : "", tracker.palmMs(),
        tracker.landmarkMs(), hs.left.valid ? "yes" : "no",
        hs.right.valid ? "yes" : "no");
    for (const Hand *h : {&hs.left, &hs.right}) {
      if (h->valid) {
        std::printf("  [score %.2f palm (%.2f,%.2f)]", h->score, h->x, h->y);
      }
    }
    std::printf("\n");
    ++total;
    bothFrames += hs.left.valid && hs.right.valid;
    anyFrames += hs.left.valid || hs.right.valid;
    for (const Hand *h : {&hs.left, &hs.right}) {
      if (h->valid) {
        scoreSum += h->score;
        ++scoreN;
      }
    }
    proc.copyTo(view);
  }

  if (view.empty()) {
    std::fprintf(stderr, "no frames processed\n");
    return 1;
  }

  drawHands(view, hs, debug);
  cv::imwrite(out, view);
  std::printf("wrote %s (%dx%d)\n", out.c_str(), view.cols, view.rows);
  std::printf("SUMMARY both=%d any=%d of %d meanScore=%.2f\n", bothFrames,
              anyFrames, total, scoreN ? scoreSum / scoreN : 0.0);

  return (hs.left.valid || hs.right.valid) ? 0 : 3; // exit 3 = no hand found
}
