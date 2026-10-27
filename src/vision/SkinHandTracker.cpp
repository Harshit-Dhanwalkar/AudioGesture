#include "audiogesture/vision/GestureTracker.h"

#include <algorithm>
#include <iostream>
#include <string>

SkinHandTracker::SkinHandTracker(int camIndex, int w, int h) : cap_(camIndex) {
  cap_.set(cv::CAP_PROP_FRAME_WIDTH, w);
  cap_.set(cv::CAP_PROP_FRAME_HEIGHT, h);
  for (const char *p :
       {"haarcascade_frontalface_default.xml",
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml"}) {
    if (face_.load(p)) {
      haveFace_ = true;
      break;
    }
  }
}

bool SkinHandTracker::poll(HandState &out) {
  cv::Mat raw;
  if (!cap_.read(raw) || raw.empty()) {
    return false;
  }
  cv::flip(raw, frame_,
           1); // mirror so moving your right hand moves right on screen

  // Skin segmentation in YCrCb (more lighting-robust than RGB)
  cv::Mat ycrcb;
  cv::cvtColor(frame_, ycrcb, cv::COLOR_BGR2YCrCb);
  cv::inRange(ycrcb, cv::Scalar(0, 133, 77), cv::Scalar(255, 173, 127), mask_);

  // Remove face (and neck) so only two hands remain
  if (haveFace_ && (frameNo_++ % 10 == 0)) {
    cv::Mat gray;
    cv::cvtColor(frame_, gray, cv::COLOR_BGR2GRAY);
    face_.detectMultiScale(gray, faces_, 1.2, 5, 0, cv::Size(80, 80));
  }

  const cv::Rect bounds(0, 0, frame_.cols, frame_.rows);
  for (const auto &f : faces_) {
    cv::Rect r(f.x - f.width * 3 / 10, f.y - f.height * 4 / 10,
               f.width * 16 / 10, f.height * 26 / 10);
    cv::rectangle(mask_, r & bounds, cv::Scalar(0), cv::FILLED);
  }

  // Clean up the mask
  cv::medianBlur(mask_, mask_, 5);
  cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, {7, 7});
  cv::morphologyEx(mask_, mask_, cv::MORPH_OPEN, k);
  cv::dilate(mask_, mask_, k);

  // Two largest blobs = hands. Centroid via image moments
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask_, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  const double minArea = 0.005 * frame_.cols * frame_.rows;
  std::vector<std::pair<double, cv::Point2f>> blobs;
  for (const auto &c : contours) {
    double a = cv::contourArea(c);
    if (a < minArea) {
      continue;
    }

    cv::Moments m = cv::moments(c);
    if (m.m00 <= 0) {
      continue;
    }

    blobs.push_back({a, {float(m.m10 / m.m00), float(m.m01 / m.m00)}});
    cv::polylines(frame_, c, true, cv::Scalar(0, 255, 0), 1);
  }

  std::sort(blobs.begin(), blobs.end(),
            [](auto &a, auto &b) { return a.first > b.first; });
  if (blobs.size() > 2) {
    blobs.resize(2);
  }

  out = HandState{};
  out.aspect = float(frame_.cols) / float(frame_.rows);
  auto norm = [&](cv::Point2f p) {
    Hand h;
    h.valid = true;
    h.x = p.x / frame_.cols;
    h.y = p.y / frame_.rows;
    return h;
  };

  if (blobs.size() == 2) {
    cv::Point2f a = blobs[0].second, b = blobs[1].second;
    if (a.x > b.x) {
      std::swap(a, b);
    }
    out.left = norm(a);
    out.right = norm(b);
  } else if (blobs.size() == 1) {
    out.right =
        norm(blobs[0].second); // single hand drives pitch + direction only
  }

  return true;
}
