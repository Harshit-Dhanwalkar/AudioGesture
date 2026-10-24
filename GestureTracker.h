#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

// Hand positions normalized to [0,1] (x right, y DOWN), image is mirrored so
// "left" is the user's left hand as seen on screen.
struct Hand {
    bool  valid = false;
    float x = 0.5f, y = 0.5f;
};
struct HandState {
    Hand left, right;
    float aspect = 4.f / 3.f; // frame width/height, needed for a true Euclidean distance
};

// Abstraction point: swap in a MediaPipe-based source by implementing poll().
class IHandSource {
public:
    virtual ~IHandSource() = default;
    virtual bool poll(HandState& out) = 0; // false = camera/stream ended
};

// Simple skin-colour + contour tracker (YCrCb threshold, face masked out via Haar cascade if found).
class SkinHandTracker : public IHandSource {
public:
    explicit SkinHandTracker(int camIndex = 0, int w = 640, int h = 480);
    bool isOpen() const { return cap_.isOpened(); }
    bool poll(HandState& out) override;
    const cv::Mat& frame() const { return frame_; } // mirrored frame with debug contours

private:
    cv::VideoCapture cap_;
    cv::Mat frame_, mask_;
    cv::CascadeClassifier face_;
    bool haveFace_ = false;
    std::vector<cv::Rect> faces_;
    int frameNo_ = 0;
};
