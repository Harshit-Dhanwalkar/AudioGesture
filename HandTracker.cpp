#include "HandTracker.h"
#include "GestureTracker.h"

#include <cstddef>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

#include <opencv2/core/utils/logger.hpp>

namespace {

constexpr int kPalmIn = 192; // palm detector input (square)
constexpr int kHandIn = 224; // landmark model input (square)
constexpr float kPi = 3.14159265358979f;

const std::vector<std::string> kLandmarkOuts = {
    "Identity",   // [0] 21x(xyz) landmarks, crop pixels
    "Identity_1", // [1] handedness (sigmoid, ~0 or ~1)
    "Identity_2", // [2] presence (0..1)   <- the score we want
    "Identity_3", // [3] 21x(xyz) world landmarks, metres
};

// HAND_DEBUG read once at startup
const bool kDebug = std::getenv("HAND_DEBUG") != nullptr;
const bool kDebugRoi = std::getenv("HAND_DEBUG_ROI") != nullptr;

double nowMs() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

bool fileExists(const std::string &p) { return std::ifstream(p).good(); }

// SSD anchors of MediaPipe palm detector.Fixed-size anchors, so only cell
// centres matter: a 24x24 grid with 2 anchors per cell (stride 8) followed by a
// 12x12 grid with 6 per cell (stride 16) = 1152 + 864 = 2016. Row-major, centre
// of cell (x,y) = ((x+.5)/n, (y+.5)/n)
std::vector<cv::Point2f> makeAnchors() {
  std::vector<cv::Point2f> a;
  a.reserve(2016);

  const int grid[2] = {24, 12}, per[2] = {2, 6};
  for (int l = 0; l < 2; ++l) {
    for (int y = 0; y < grid[l]; ++y) {
      for (int x = 0; x < grid[l]; ++x) {
        for (int k = 0; k < per[l]; ++k) {
          a.emplace_back((x + 0.5f) / grid[l], (y + 0.5f) / grid[l]);
        }
      }
    }
  }

  return a;
}

// OpenCV older than 4.7 can't parse the stock landmark model (see
// models/make_opencv_compat.py), so on those versions we try the "_cvcompat"
// file first
cv::dnn::Net loadNet(const std::vector<std::string> &candidates,
                     std::string &used, std::string &err) {
  using namespace cv::utils::logging;
  const LogLevel prev = getLogLevel();
  setLogLevel(LOG_LEVEL_SILENT); // the importer is very noisy on failure
  cv::dnn::Net net;
  for (const std::string &p : candidates) {
    if (!fileExists(p)) {
      continue;
    }

    try {
      net = cv::dnn::readNetFromONNX(p);
      if (!net.empty()) {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        used = p;
        setLogLevel(prev);
        return net;
      }
    } catch (const cv::Exception &e) {
      err = "OpenCV " CV_VERSION " could not load " + p + ": " +
            e.msg.substr(0, e.msg.find('\n'));
    }
  }

  setLogLevel(prev);
  if (err.empty()) {
    err = "model file not found (tried: " +
          (candidates.empty() ? std::string("-") : candidates.front()) + ")";
  }

  return cv::dnn::Net();
}

// Hand "up" direction in image coordinates for a crop rotated by `angle`
inline cv::Point2f upVec(float angle) {
  return {std::sin(angle), -std::cos(angle)};
}

inline cv::Point2f rightVec(float angle) {
  return {std::cos(angle), std::sin(angle)};
}

// Rotation that makes vector wrist->middle-finger point "up" in crop angle =
// \atan2(dx, -dy): 0 when fingers point up image, +90deg when they point right
inline float handAngle(cv::Point2f wrist, cv::Point2f middle) {
  return std::atan2(middle.x - wrist.x, -(middle.y - wrist.y));
}

inline float dist(cv::Point2f a, cv::Point2f b) {
  return std::hypot(a.x - b.x, a.y - b.y);
}

} // namespace

// Models dir
std::string findModelsDir(const std::string &hint) {
  std::vector<std::string> cands;
  if (!hint.empty()) {
    cands.push_back(hint);
  }

  cands.push_back("models");
  char buf[4096];
  const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (n > 0) {
    buf[n] = 0;
    const std::string exe(buf);
    const std::string dir = exe.substr(0, exe.find_last_of('/'));
    cands.push_back(dir + "/../models");
    cands.push_back(dir + "/models");
  }

  for (const auto &c : cands) {
    if (fileExists(c + "/" + kPalmModelFile))
      return c;
  }

  return hint.empty() ? "models" : hint;
}

// Tracker
LandmarkHandTracker::LandmarkHandTracker(const HandTrackerConfig &cfg)
    : cfg_(cfg), anchors_(makeAnchors()) {
  cfg_.maxHands = std::clamp(cfg_.maxHands, 1, 2);

  std::string used, err;
  palmNet_ = loadNet({cfg_.palmModel}, used, err);
  if (palmNet_.empty()) {
    error_ = "palm detector: " + err;
    return;
  }

  std::string handPath = cfg_.landmarkModel;
  std::string compat = handPath;
  const auto dot = compat.rfind(".onnx");
  if (dot != std::string::npos) {
    compat.insert(dot, "_cvcompat");
  }

  const bool oldCv = (CV_VERSION_MAJOR * 100 + CV_VERSION_MINOR) < 407;
  err.clear();
  handNet_ = loadNet(oldCv ? std::vector<std::string>{compat, handPath}
                           : std::vector<std::string>{handPath, compat},
                     handModelUsed_, err);
  if (handNet_.empty()) {
    error_ = "hand landmark model: " + err;
    if (oldCv) {
      error_ += "\n  (OpenCV < 4.7 needs patched model: run `make models` "
                "with `pip install onnx` available)";
    }

    return;
  }

  palmOutNames_ = palmNet_.getUnconnectedOutLayersNames();
  handOutNames_ = handNet_.getUnconnectedOutLayersNames();
  if (handOutNames_.size() != 4) {
    error_ = "unexpected landmark model outputs";
    return;
  }

  // Verify canonical output names are all present so name-pinned forward()
  // below can't silently fall back to an unexpected order
  for (const std::string &want : kLandmarkOuts) {
    if (std::find(handOutNames_.begin(), handOutNames_.end(), want) ==
        handOutNames_.end()) {
      error_ = "landmark model is missing output '" + want + "'; got:";
      for (const auto &n : handOutNames_) {
        error_ += " '" + n + "'";
      }

      if (kDebug) {
        std::fprintf(stderr, "%s\n", error_.c_str());
      }
      return;
    }
  }

  for (auto &t : tracks_) {
    for (auto &f : t.filt) {
      f = OneEuroFilter(cfg_.filterMinCutoff, cfg_.filterBeta);
    }
  }

  ok_ = true;
}

// Stage 1

std::vector<LandmarkHandTracker::Palm>
LandmarkHandTracker::detectPalms(const cv::Mat &bgr) {
  const double t0 = nowMs();
  const int W = bgr.cols;
  const int H = bgr.rows;

  // Letterbox into 192x192 (keep aspect ratio, pad with black)
  const float ratio = std::min(float(kPalmIn) / W, float(kPalmIn) / H);
  const cv::Size rs(std::max(1, int(W * ratio)), std::max(1, int(H * ratio)));
  cv::resize(bgr, tmp_, rs, 0, 0,
             cv::INTER_AREA); // area averaging: shrinks AND denoises (bilinear
                              // would alias)
  const int padW = kPalmIn - rs.width, padH = kPalmIn - rs.height;
  const int left = padW / 2, top = padH / 2;
  cv::copyMakeBorder(tmp_, tmp_, top, padH - top, left, padW - left,
                     cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
  cv::cvtColor(tmp_, tmp_, cv::COLOR_BGR2RGB);
  tmp_.convertTo(tmpF_, CV_32FC3, 1.0 / 255.0);

  // Model wants NHWC: a float HxWx3 image is already laid out that way in
  // memory
  const int sz[4] = {1, kPalmIn, kPalmIn, 3};
  palmNet_.setInput(cv::Mat(4, sz, CV_32F, tmpF_.data));
  palmNet_.forward(palmOuts_, palmOutNames_);

  const cv::Mat *boxes = nullptr, *scores = nullptr;
  for (const cv::Mat &m : palmOuts_) {
    if (m.dims == 3 && m.size[2] == 18) {
      boxes = &m;
    } else if (m.dims == 3 && m.size[2] == 1) {
      scores = &m;
    }
  }

  std::vector<Palm> result;
  if (!boxes || !scores || boxes->size[1] != int(anchors_.size())) {
    palmMs_ = nowMs() - t0;
    return result;
  }

  // Network outputs are offsets from anchor, in units of 192 px input:
  //   centre = anchor + d/192,  size = wh/192,  keypoint = anchor + k/192 (all
  //   in 0..1 of letterboxed square)
  // Map back to source pixels: square covers max(W,H) source pixels, minus
  // padding
  const float scale = float(std::max(W, H));
  const float padX = left / ratio, padY = top / ratio;
  const float *b = boxes->ptr<float>();
  const float *s = scores->ptr<float>();
  const float logitThr = std::log(
      cfg_.palmScore / (1.f - cfg_.palmScore)); // compare logits, skip sigmoid

  std::vector<cv::Rect2d> rects;
  std::vector<float> confs;
  std::vector<int> ids;
  for (size_t i = 0; i < anchors_.size(); ++i) {
    if (s[i] < logitThr) {
      continue;
    }

    const float *r = b + i * 18;
    const cv::Point2f a = anchors_[i];
    const float cx = (r[0] / kPalmIn + a.x) * scale - padX;
    const float cy = (r[1] / kPalmIn + a.y) * scale - padY;
    const float w = r[2] / kPalmIn * scale, h = r[3] / kPalmIn * scale;
    rects.emplace_back(cx - w / 2, cy - h / 2, w, h);
    confs.push_back(1.f / (1.f + std::exp(-std::clamp(s[i], -50.f, 50.f))));
    ids.push_back(int(i));
  }

  std::vector<int> keep;
  cv::dnn::NMSBoxes(rects, confs, cfg_.palmScore, cfg_.palmNms,
                    keep); // returns best-first

  for (int k : keep) {
    const float *r = b + size_t(ids[k]) * 18;
    const cv::Point2f a = anchors_[ids[k]];
    Palm p;
    p.box = cv::Rect2f(float(rects[k].x), float(rects[k].y),
                       float(rects[k].width), float(rects[k].height));
    p.score = confs[k];
    for (int j = 0; j < 7; ++j) {
      p.kp[j] = {(r[4 + 2 * j] / kPalmIn + a.x) * scale - padX,
                 (r[5 + 2 * j] / kPalmIn + a.y) * scale - padY};
    }

    result.push_back(p);
  }

  if (kDebug) {
    std::fprintf(stderr, "[palm] %zu candidates\n", result.size());
    for (size_t i = 0; i < std::min(result.size(), size_t(3)); ++i) {
      std::fprintf(stderr, "[palm] #%zu score=%.3f box=(%.0f,%.0f %.0fx%.0f)\n",
                   i, result[i].score, result[i].box.x, result[i].box.y,
                   result[i].box.width, result[i].box.height);
    }
  }

  palmMs_ = nowMs() - t0;
  return result;
}

// Palm detection -> crop for stage 2 (MediaPipe's palm_detection -> hand ROI
// rules):
//  * rotate so wrist (kp0) -> middle-finger base (kp2) points up
//  * move window 0.5 box-heights toward fingers (box covers palm only)
//  * make it square on longer side and enlarge x2.6 so all fingers fit
LandmarkHandTracker::Roi LandmarkHandTracker::roiFromPalm(const Palm &p) {
  Roi r;
  r.angle = handAngle(p.kp[0], p.kp[2]);
  const cv::Point2f centre(p.box.x + p.box.width / 2,
                           p.box.y + p.box.height / 2);
  r.c = centre + upVec(r.angle) * (0.5f * p.box.height);
  r.size = std::max(p.box.width, p.box.height) * 2.6f;
  return r;
}

// Joints from this frame -> crop for next frame (MediaPipe's landmarks -> ROI
// rules): bounding box of joints measured in hand's own (rotated) frame,
// shifted 0.1 heights toward fingertips, squared and enlarged x2
//
// Joints are clamped into frame first.  A hand that pokes out of image gets its
// hidden fingers "hallucinated" outside it; letting those phantom points steer
// next crop made ROI creep outward frame after frame until hand was lost
LandmarkHandTracker::Roi LandmarkHandTracker::roiFromLandmarks(
    const std::array<cv::Point2f, kNumLandmarks> &px, int W, int H) {
  Roi r;
  r.angle = handAngle(px[0], px[9]);
  const cv::Point2f R = rightVec(r.angle), U = upVec(r.angle);
  auto inFrame = [&](cv::Point2f p) {
    return cv::Point2f(std::clamp(p.x, 0.f, W - 1.f),
                       std::clamp(p.y, 0.f, H - 1.f));
  };

  const cv::Point2f wrist = inFrame(px[0]);
  float minU = 1e9f, maxU = -1e9f, minV = 1e9f, maxV = -1e9f;
  for (const auto &p : px) {
    const cv::Point2f d = inFrame(p) - wrist;
    const float u = d.x * R.x + d.y * R.y, v = d.x * U.x + d.y * U.y;

    minU = std::min(minU, u);
    maxU = std::max(maxU, u);
    minV = std::min(minV, v);
    maxV = std::max(maxV, v);
  }

  const float w = maxU - minU, h = maxV - minV;
  r.c = wrist + R * ((minU + maxU) / 2) + U * ((minV + maxV) / 2 + 0.1f * h);
  r.size = std::max(w, h) * 2.0f;
  return r;
}

// Stage 2

bool LandmarkHandTracker::runLandmarks(
    const cv::Mat &bgr, const Roi &roi,
    std::array<cv::Point2f, kNumLandmarks> &px, float &score) {
  if (!(roi.size > 8.f) || !std::isfinite(roi.c.x) || !std::isfinite(roi.c.y)) {
    return false;
  }

  const double t0 = nowMs();

  // Affine map crop pixel (u,v) -> source pixel: p = C + R * k * ((u,v) - 112)
  // (k = source pixels per crop pixel). warpAffine with WARP_INVERSE_MAP takes
  // exactly this dst->src matrix and does rotate + scale + crop in one
  // resampling step
  const float k = roi.size / kHandIn;
  const float c = std::cos(roi.angle);
  const float s = std::sin(roi.angle);
  const float h = kHandIn * 0.5f;
  const cv::Matx23f A(c * k, -s * k, roi.c.x - c * k * h + s * k * h, s * k,
                      c * k, roi.c.y - s * k * h - c * k * h);
  cv::warpAffine(bgr, crop_, A, cv::Size(kHandIn, kHandIn),
                 cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT,
                 cv::Scalar(0, 0, 0));
  cv::cvtColor(crop_, crop_, cv::COLOR_BGR2RGB);
  crop_.convertTo(cropF_, CV_32FC3, 1.0 / 255.0);

  if (kDebugRoi) {
    std::fprintf(stderr, "[roi] c=(%.1f,%.1f) size=%.1f angle=%.1fdeg\n",
                 roi.c.x, roi.c.y, roi.size, roi.angle * 180.0f / kPi);
  }

  // Model wants NHWC: a float HxWx3 image is already laid out that way in
  // memory. Forward by name so handOuts_[i] maps to a known tensor on any
  // OpenCV version (getUnconnectedOutLayersNames order is graph-traversal
  // dependent - on 4.5.4 it lands at {Identity, Identity_3, Identity_2,
  // Identity_1}).
  const int sz[4] = {1, kHandIn, kHandIn, 3};
  handNet_.setInput(cv::Mat(4, sz, CV_32F, cropF_.data));
  if (useNamedOuts_) {
    handNet_.forward(handOuts_, kLandmarkOuts);
  } else {
    handNet_.forward(handOuts_, handOutNames_);
  }

  if (kDebug) {
    std::fprintf(stderr, "[lm] outs=%zu", handOuts_.size());
    for (size_t i = 0; i < handOuts_.size(); ++i) {
      std::fprintf(
          stderr, " [%zu:total=%d first=%.4f]", i, (int)handOuts_[i].total(),
          handOuts_[i].total() > 0 ? handOuts_[i].ptr<float>()[0] : 0.f);
    }
    std::fprintf(stderr, "\n");
  }

  landmarkMs_ += nowMs() - t0;

  // Outputs when forwarded by name (kLandmarkOuts order):
  //   [0] Identity    = 21x(xyz) landmarks in crop pixels
  //   [1] Identity_1  = handedness (sigmoid, saturates near 0 or 1)
  //   [2] Identity_2  = presence score (0..1)  <- value we track on
  //   [3] Identity_3  = 21x(xyz) world landmarks (unused)
  // Fallback order on OpenCV 4.5.4 (getUnconnectedOutLayersNames):
  //   [0]=lm, [1]=world, [2]=presence, [3]=handed
  // Index 2 is presence in both - coincidence, but name-based path pins it on
  // any future OpenCV that reorders
  const int lmIdx = 0;
  const int scoreIdx = 2;

  if (handOuts_.size() <= size_t(scoreIdx) || handOuts_[lmIdx].total() < 63 ||
      handOuts_[scoreIdx].total() != 1) {
    return false;
  }

  // Sanity check: if forward() silently gave world-landmark tensor in slot 0,
  // landmark magnitudes would be < 1 m instead of > 10 px
  if (kDebug) {
    double mx = 0;
    cv::Mat mag = cv::abs(handOuts_[lmIdx].reshape(1, 1));
    cv::minMaxLoc(mag, nullptr, &mx);
    if (mx < 2.0) {
      std::fprintf(stderr,
                   "[lm] output 0 max=%g looks like world coords - output "
                   "order is wrong (useNamedOuts_=%d)\n",
                   mx, (int)useNamedOuts_);
    }
  }

  const float *lm = handOuts_[lmIdx].ptr<float>();
  score = handOuts_[scoreIdx].ptr<float>()[0];
  for (int i = 0; i < kNumLandmarks; ++i) {
    const float x = lm[3 * i], y = lm[3 * i + 1];
    px[i] = {A(0, 0) * x + A(0, 1) * y + A(0, 2),
             A(1, 0) * x + A(1, 1) * y + A(1, 2)};
  }

  return std::isfinite(score);
}

cv::Point2f LandmarkHandTracker::palmCentre(
    const std::array<cv::Point2f, kNumLandmarks> &px) {
  // Wrist + four finger bases: a rigid patch of hand that doesn't move when you
  // curl your fingers, so it's a much steadier control point than a blob
  // centroid
  static const int ids[5] = {0, 5, 9, 13, 17};
  cv::Point2f c(0, 0);
  for (int i : ids) {
    c += px[i];
  }

  return c * 0.2f;
}

bool LandmarkHandTracker::overlapsTrack(const Roi &r) const {
  for (const Track &t : tracks_) {
    if (t.active && dist(t.roi.c, r.c) < 0.25f * std::max(t.roi.size, r.size)) {
      return true;
    }
  }

  return false;
}

LandmarkHandTracker::Track *LandmarkHandTracker::freeTrack() {
  for (Track &t : tracks_) {
    if (!t.active) {
      return &t;
    }
  }

  return nullptr;
}

// Per-frame logic

void LandmarkHandTracker::process(const cv::Mat &bgr, double dt,
                                  HandState &out) {
  out = HandState{};
  if (!ok_ || bgr.empty()) {
    return;
  }

  const int W = bgr.cols, H = bgr.rows;
  out.aspect = float(W) / float(H);
  dt = std::clamp(dt, 1e-3, 0.25);
  landmarkMs_ = 0;

  // Follow hands: run landmark model on last frame's ROI
  bool lostOne = false;
  for (Track &t : tracks_) {
    if (!t.active) {
      continue;
    }

    std::array<cv::Point2f, kNumLandmarks> px;
    float score = 0.f;
    if (runLandmarks(bgr, t.roi, px, score) && score >= cfg_.presenceKeep) {
      t.raw = px;
      t.score = score;
      t.roi = roiFromLandmarks(px, W, H);
      ++t.age;
    } else {
      t.active = false;
      lostOne = true;
    }
  }

  // If a hand is missing, run palm detector: every frame if none is tracked,
  // immediately
  //    when one was just lost (so it is re-acquired without a gap), otherwise
  //    only every few frames so a second hand is picked up without paying for
  //    detector constantly
  int nActive = 0;
  for (const Track &t : tracks_) {
    nActive += t.active;
  }

  palmMs_ = 0;
  if (nActive < cfg_.maxHands &&
      (nActive == 0 || lostOne || ++sinceDetect_ >= cfg_.redetectEvery)) {
    sinceDetect_ = 0;
    for (const Palm &palm : detectPalms(bgr)) {
      if (nActive >= cfg_.maxHands) {
        break;
      }

      const Roi roi = roiFromPalm(palm);
      if (overlapsTrack(roi)) {
        continue; // that's a hand already being followed
      }

      std::array<cv::Point2f, kNumLandmarks> px;
      float score = 0.f;
      if (!runLandmarks(bgr, roi, px, score) || score < cfg_.presenceStart) {
        continue;
      }

      Track *t = freeTrack();
      if (!t) {
        break;
      }

      t->active = true;
      t->fresh = true;
      t->age = 0;
      t->score = score;
      t->raw = px;
      t->roi = roiFromLandmarks(px, W, H);
      for (auto &f : t->filt) {
        f.reset();
      }

      ++nActive;
    }
  }

  // Two tracks that slid onto same hand: keep more confident one
  if (nActive == 2) {
    Track &a = tracks_[0], &b = tracks_[1];
    if (dist(palmCentre(a.raw), palmCentre(b.raw)) <
        0.25f * std::min(a.roi.size, b.roi.size)) {
      (a.score >= b.score ? b : a).active = false;
      --nActive;
    }
  }

  // Left/right slots
  // A lone hand is "right" (drives pitch/direction). When a second hand
  //    shows up, one further left on screen becomes "left"; after that each
  //    hand keeps its slot even if they cross
  const bool anyFresh = tracks_[0].fresh || tracks_[1].fresh;
  if (nActive == 2 && anyFresh) {
    const bool zeroIsLeft =
        palmCentre(tracks_[0].raw).x <= palmCentre(tracks_[1].raw).x;
    tracks_[0].slot = zeroIsLeft ? 0 : 1;
    tracks_[1].slot = zeroIsLeft ? 1 : 0;
  } else if (nActive == 1) {
    for (Track &t : tracks_) {
      if (t.active && t.fresh) {
        t.slot = 1;
      }
    }
  }

  // 5) Smooth and publish.
  for (Track &t : tracks_) {
    t.fresh = false;
    if (!t.active) {
      continue;
    }

    for (int i = 0; i < kNumLandmarks; ++i) {
      t.smooth[i] = {t.filt[2 * i](t.raw[i].x, float(dt)),
                     t.filt[2 * i + 1](t.raw[i].y, float(dt))};
    }

    Hand h;
    h.valid = h.hasSkeleton = true;
    h.score = t.score;
    const cv::Point2f pc = palmCentre(t.smooth);
    h.x = pc.x / W;
    h.y = pc.y / H;
    for (int i = 0; i < kNumLandmarks; ++i) {
      h.lm[i] = {t.smooth[i].x / W, t.smooth[i].y / H};
    }
    h.roiCenter = t.roi.c;
    h.roiSize = t.roi.size;
    h.roiAngle = t.roi.angle;
    (t.slot == 0 ? out.left : out.right) = h;
  }
}

// Camera source
CameraHandSource::CameraHandSource(int camIndex, const HandTrackerConfig &cfg,
                                   int w, int h)
    : cap_(camIndex), tracker_(cfg) {
  cap_.set(cv::CAP_PROP_FRAME_WIDTH, w);
  cap_.set(cv::CAP_PROP_FRAME_HEIGHT, h);
  cap_.set(cv::CAP_PROP_FPS, 30);
}

CameraHandSource::CameraHandSource(const std::string &videoFile,
                                   const HandTrackerConfig &cfg)
    : cap_(videoFile), tracker_(cfg), isFile_(true) {}

bool CameraHandSource::poll(HandState &out) {
  if (!cap_.read(raw_) || raw_.empty()) {
    if (!isFile_) {
      return false;
    }

    cap_.set(cv::CAP_PROP_POS_FRAMES, 0); // loop video files
    if (!cap_.read(raw_) || raw_.empty()) {
      return false;
    }
  }

  if (isFile_) {
    mirrored_ = raw_;
  } else {
    cv::flip(raw_, mirrored_, 1); // mirror webcams so it feels like a mirror
  }

  const cv::Mat &proc = enh_.apply(mirrored_); // brightened if the room is dark

  const double t = nowMs() / 1000.0;
  const double dt = (lastT_ > 0) ? t - lastT_ : 1.0 / 30.0;
  lastT_ = t;
  fps_ = (fps_ == 0) ? 1.0 / dt : 0.9 * fps_ + 0.1 / dt;

  tracker_.process(proc, dt, out);
  proc.copyTo(view_);

  return true;
}

// Drawing
void drawHands(cv::Mat &img, const HandState &hs, bool debugRoi) {
  // Bones as (parent, child).  Landmark ids: 0 wrist; 1-4 thumb; 5-8 index;
  // 9-12 middle; 13-16 ring; 17-20 pinky.
  static const int bones[][2] = {
      {0, 1},  {1, 2},   {2, 3},   {3, 4},   {0, 5},   {5, 6},   {6, 7},
      {7, 8},  {9, 10},  {10, 11}, {11, 12}, {13, 14}, {14, 15}, {15, 16},
      {0, 17}, {17, 18}, {18, 19}, {19, 20}, {5, 9},   {9, 13},  {13, 17}};
  static const cv::Scalar fingerColour[5] = {
      {0, 165, 255},
      {90, 220, 90},
      {255, 200, 40},
      {255, 110, 70},
      {220, 90, 255}}; // BGR: thumb..pinky
  static const cv::Scalar palmColour(235, 235, 235);

  auto finger = [](int i) { return i == 0 ? -1 : (i - 1) / 4; };
  auto px = [&](const Hand &h, int i) {
    return cv::Point(int(std::lround(h.lm[i].x * img.cols)),
                     int(std::lround(h.lm[i].y * img.rows)));
  };

  for (const Hand *hp : {&hs.left, &hs.right}) {
    const Hand &h = *hp;
    if (!h.valid || !h.hasSkeleton) {
      continue;
    }

    if (debugRoi) {
      const cv::Point2f R =
          cv::Point2f(std::cos(h.roiAngle), std::sin(h.roiAngle)) *
          (h.roiSize / 2);
      const cv::Point2f U =
          cv::Point2f(std::sin(h.roiAngle), -std::cos(h.roiAngle)) *
          (h.roiSize / 2);
      const cv::Point2f c = h.roiCenter;
      const cv::Point2f q[4] = {c - R + U, c + R + U, c + R - U, c - R - U};
      for (int i = 0; i < 4; ++i)
        cv::line(img, q[i], q[(i + 1) % 4],
                 i == 0 ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 180, 180), 1,
                 cv::LINE_AA);
    }

    for (const auto &b : bones) {
      // A bone takes its finger's colour when both ends are on the same finger
      // (or it is the wrist->thumb bone); palm outline bones are white.
      const int fa = finger(b[0]), fb = finger(b[1]);
      const bool onFinger = (fa == fb && fa >= 0) || (b[0] == 0 && b[1] == 1);
      const cv::Scalar col = onFinger ? fingerColour[fb] : palmColour;
      cv::line(img, px(h, b[0]), px(h, b[1]), cv::Scalar(0, 0, 0), 4,
               cv::LINE_AA); // dark outline for contrast
      cv::line(img, px(h, b[0]), px(h, b[1]), col, 2, cv::LINE_AA);
    }

    for (int i = 0; i < kNumLandmarks; ++i) {
      const bool tip = i > 0 && i % 4 == 0;
      const int r = (i == 0) ? 6 : tip ? 5 : 3;
      cv::circle(img, px(h, i), r + 1, cv::Scalar(0, 0, 0), cv::FILLED,
                 cv::LINE_AA);
      cv::circle(img, px(h, i), r,
                 i == 0 ? palmColour : fingerColour[std::max(0, finger(i))],
                 cv::FILLED, cv::LINE_AA);
    }

    char buf[24];
    std::snprintf(buf, sizeof buf, "%s %.2f", (hp == &hs.left) ? "L" : "R",
                  h.score);
    const cv::Point w = px(h, 0);
    cv::putText(img, buf, {w.x - 18, w.y + 22}, cv::FONT_HERSHEY_SIMPLEX, 0.45,
                {0, 0, 0}, 3, cv::LINE_AA);
    cv::putText(img, buf, {w.x - 18, w.y + 22}, cv::FONT_HERSHEY_SIMPLEX, 0.45,
                {255, 255, 255}, 1, cv::LINE_AA);
  }
}
