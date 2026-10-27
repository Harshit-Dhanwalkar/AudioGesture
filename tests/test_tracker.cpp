// Landmark tracker: models load on this OpenCV, nothing is hallucinated on empty
// or noise frames, and (if tests/data/hand.jpg exists) a real hand is found.
// Needs the ONNX files (`make models`); exits 77 (= SKIP for ctest) without them.
#include <cstdio>
#include <fstream>

#include "audiogesture/vision/HandTracker.h"
#include "check.h"

int main() {
  const std::string dir = findModelsDir("");
  if (!std::ifstream(dir + "/" + kPalmModelFile).good()) {
    std::printf("SKIP: no models in '%s' (run `make models`)\n", dir.c_str());
    return 77;
  }

  HandTrackerConfig cfg;
  cfg.palmModel = dir + "/" + kPalmModelFile;
  cfg.landmarkModel = dir + "/" + kHandModelFile;
  LandmarkHandTracker t(cfg);
  if (!t.ok()) {
    std::fprintf(stderr, "tracker failed to load: %s\n", t.error().c_str());
    return 1;
  }
  std::printf("OpenCV %s, landmark model %s\n", CV_VERSION, t.landmarkModelUsed().c_str());

  HandState hs;
  cv::RNG rng(3);

  // Empty scenes: no phantom hands, including through the scan fallback
  cv::Mat black(480, 640, CV_8UC3, cv::Scalar::all(0));
  cv::Mat noise(480, 640, CV_8UC3);
  for (int i = 0; i < 14; ++i) {
    t.process(black, 1 / 30.0, hs);
    CHECK(!hs.left.valid && !hs.right.valid);
    rng.fill(noise, cv::RNG::UNIFORM, 0, 255);
    t.process(noise, 1 / 30.0, hs);
    CHECK(!hs.left.valid && !hs.right.valid);
  }

  // Optional real-hand fixture (not shipped: add your own photo)
  const bool haveHand = std::ifstream("tests/data/hand.jpg").good();
  const cv::Mat hand = haveHand ? cv::imread("tests/data/hand.jpg") : cv::Mat();
  if (!hand.empty()) {
    t.reset();
    bool found = false;
    for (int i = 0; i < 8 && !found; ++i) {
      t.process(hand, 1 / 30.0, hs);
      found = hs.left.valid || hs.right.valid;
    }
    CHECK(found);
    for (const Hand *h : {&hs.left, &hs.right}) {
      if (h->valid) CHECK(h->score > 0.5f && h->hasSkeleton);
    }
  }

  TEST_MAIN_END();
}
