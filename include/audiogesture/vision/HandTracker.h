#pragma once
// Landmark-based hand tracker: MediaPipe two-stage pipeline running on
// cv::dnn
//
//   Stage 1 palm detector (192x192 SSD, 2016 anchors) finds palms anywhere
//   in frame
//   Stage 2  hand landmarks (224x224 CNN) regresses 21 3-D joints inside a
//   rotated square crop around one hand
//
// Stage 2 also reports "is a hand really in this crop?".  We exploit that to
// track: joints from frame N define crop for frame N+1, so (slower) palm
// detector only runs when a hand is missing.  Result: a real skeleton (wrist +
// 4 joints per finger) instead of a skin-colour blob, which keeps working when
// colour cues die in low light
//
// Models: models/get_models.sh downloads them (see README)

#include <array>
#include <string>
#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

#include "audiogesture/vision/GestureTracker.h"
#include "audiogesture/vision/LowLightEnhancer.h"
#include "audiogesture/vision/OneEuroFilter.h"

inline constexpr const char *kPalmModelFile =
    "palm_detection_mediapipe_2023feb.onnx";
inline constexpr const char *kHandModelFile =
    "handpose_estimation_mediapipe_2023feb.onnx";

// Looks for model files in: hint, ./models, <exe>/../models, <exe>/models
std::string findModelsDir(const std::string &hint);

struct HandTrackerConfig {
  std::string palmModel; // full paths
  std::string landmarkModel;
  int maxHands = 2;        // 1 or 2
  float palmScore = 0.25f; // palm detector confidence needed to consider a
                           // candidate Deliberately low (MediaPipe uses 0.5):
                           // landmark model's "hand present" score below is
                           // real verifier, and dim, noisy frames need slack
  float palmNms = 0.3f;       // non-max-suppression IoU
  float presenceStart = 0.5f; // landmark confidence to start tracking
  float presenceKeep = 0.4f;  // lower => hysteresis, fewer drop-outs
  int redetectEvery = 2; // while a slot is free but a hand is tracked, look for
                         // other hand every N frames
  // When the palm detector finds nothing (typical in dark, noisy frames) probe a
  // few wide crops with the landmark model, which is a better judge of "is
  // there a hand" than the palm SSD. One crop per frame, cycling, until a hand
  // is acquired; then normal ROI tracking takes over.
  bool scanFallback = true;
  float scanPresence = 0.55f;
  float filterMinCutoff = 1.5f; // One-Euro smoothing of landmark pixels (Hz)
  float filterBeta = 0.02f;
};

// Camera-agnostic inference.
// Feed it BGR frames (already mirrored / enhanced it)
class LandmarkHandTracker {
public:
  explicit LandmarkHandTracker(const HandTrackerConfig &cfg);

  bool ok() const { return ok_; }
  const std::string &error() const { return error_; }
  const std::string &landmarkModelUsed() const { return handModelUsed_; }

  // dt = seconds since previous call (drives smoothing filters)
  void process(const cv::Mat &bgr, double dt, HandState &out);

  // Forget all tracks (next frame starts from palm detection / scan)
  void reset();

  // Timings of last call, for HUD
  double palmMs() const { return palmMs_; }
  double landmarkMs() const { return landmarkMs_; }
  double scanMs() const { return scanMs_; }

private:
  // A square, rotated crop window: centre, side length (px) and rotation
  // (radians) such that hand's "up" axis (wrist -> middle finger) maps to
  // crop's up axis
  struct Roi {
    cv::Point2f c{0.f, 0.f};
    float size = 0.f, angle = 0.f;
  };
  struct Palm {
    cv::Rect2f box;
    std::array<cv::Point2f, 7> kp;
    float score = 0.f;
  };
  struct Track {
    bool active = false;
    bool fresh = false; // started this frame (needs a left/right slot)
    int slot = 1;       // 0 = left, 1 = right
    int age = 0;
    float score = 0.f;
    Roi roi;                                      // crop to use on NEXT frame
    std::array<cv::Point2f, kNumLandmarks> raw{}; // pixel coords, unfiltered
    std::array<cv::Point2f, kNumLandmarks>
        smooth{}; // pixel coords, One-Euro filtered
    std::array<OneEuroFilter, 2 * kNumLandmarks> filt;
  };

  std::vector<Palm> detectPalms(const cv::Mat &bgr);
  bool runLandmarks(const cv::Mat &bgr, const Roi &roi,
                    std::array<cv::Point2f, kNumLandmarks> &px, float &score);
  static Roi roiFromPalm(const Palm &p);
  static Roi roiFromLandmarks(const std::array<cv::Point2f, kNumLandmarks> &px,
                              int W, int H);
  static cv::Point2f
  palmCentre(const std::array<cv::Point2f, kNumLandmarks> &px);
  bool overlapsTrack(const Roi &r) const;
  bool scanForHand(const cv::Mat &bgr, std::array<cv::Point2f, kNumLandmarks> &px,
                   float &score);
  Track *freeTrack();

  HandTrackerConfig cfg_;
  bool ok_ = false;
  std::string error_, handModelUsed_;
  bool useNamedOuts_ = true;

  cv::dnn::Net palmNet_, handNet_;
  std::vector<std::string> palmOutNames_, handOutNames_;
  std::vector<cv::Mat> palmOuts_, handOuts_;
  std::vector<cv::Point2f> anchors_;
  cv::Mat tmp_, tmpF_, crop_, cropF_;

  std::array<Track, 2> tracks_;
  int sinceDetect_ = 0;
  int scanIdx_ = 0;
  double palmMs_ = 0, landmarkMs_ = 0, scanMs_ = 0;
};

// Webcam (or video file) + low-light enhancement + landmark tracking, as an
// IHandSource
class CameraHandSource : public IHandSource {
public:
  CameraHandSource(int camIndex, const HandTrackerConfig &cfg, int w = 640,
                   int h = 480);
  CameraHandSource(const std::string &videoFile, const HandTrackerConfig &cfg);

  bool isOpen() const { return cap_.isOpened(); }
  bool modelsOk() const { return tracker_.ok(); }
  const std::string &error() const { return tracker_.error(); }

  bool poll(HandState &out) override;
  const cv::Mat &frame() const override {
    return view_;
  } // enhanced (if active), mirrored, no overlay

  LowLightEnhancer &enhancer() { return enh_; }
  const LowLightEnhancer &enhancer() const { return enh_; }
  const LandmarkHandTracker &tracker() const { return tracker_; }
  double fps() const { return fps_; }

private:
  cv::VideoCapture cap_;
  LandmarkHandTracker tracker_;
  LowLightEnhancer enh_;
  bool isFile_ = false;
  cv::Mat raw_, mirrored_, view_;
  double lastT_ = 0, fps_ = 0;
};

// Draws 21-joint skeleton (per-finger colours) and, optionally, tracking ROI
void drawHands(cv::Mat &img, const HandState &hs, bool debugRoi);
