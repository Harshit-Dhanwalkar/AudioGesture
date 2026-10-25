#pragma once
#include <array>
#include <opencv2/opencv.hpp>
#include <vector>

constexpr int kNumLandmarks =
    21; // MediaPipe hand model: 0 wrist, 1-4 thumb, 5-8 index, 9-12 middle,
        // 13-16 ring, 17-20 pinky

// Hand positions normalized to [0,1] (x right, y DOWN), image is mirrored so
// "left" is the user's left hand as seen on screen
struct Hand {
  bool valid = false;
  float x = 0.5f, y = 0.5f; // palm centre

  // Filled only by the landmark tracker (HandTracker.h):
  bool hasSkeleton = false;
  float score = 0.f; // landmark-model "hand present" confidence, 0..1
  std::array<cv::Point2f, kNumLandmarks>
      lm{};                        // normalized image coords (smoothed)
  cv::Point2f roiCenter{0.f, 0.f}; // tracking ROI, in pixels (debug overlay)
  float roiSize = 0.f, roiAngle = 0.f;
};

struct HandState {
  Hand left, right;
  float aspect =
      4.f / 3.f; // frame width/height, needed for a true Euclidean distance
};

// Anything that can turn camera frames into HandState: swap implementations
// freely
class IHandSource {
public:
  virtual ~IHandSource() = default;
  virtual bool poll(HandState &out) = 0;    // false = camera/stream ended
  virtual const cv::Mat &frame() const = 0; // frame to show as the preview
};

// Simple skin-colour + contour tracker (YCrCb threshold, face masked out via
// Haar cascade if found). Fallback when the ONNX models aren't available; poor
// in low light
class SkinHandTracker : public IHandSource {
public:
  explicit SkinHandTracker(int camIndex = 0, int w = 640, int h = 480);
  bool isOpen() const { return cap_.isOpened(); }
  bool poll(HandState &out) override;
  const cv::Mat &frame() const override {
    return frame_;
  } // mirrored frame with debug contours

private:
  cv::VideoCapture cap_;
  cv::Mat frame_, mask_;
  cv::CascadeClassifier face_;
  bool haveFace_ = false;
  std::vector<cv::Rect> faces_;
  int frameNo_ = 0;
};
