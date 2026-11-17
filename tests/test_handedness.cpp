// HandednessResolver: learns model's polarity from clearly separated hands, then settles ambiguous
// cases (crossed hands, a lone hand on "wrong" half)

#include "audiogesture/vision/Handedness.h"
#include "check.h"

namespace {
// Model
// convention A: handedness high (>0.5) = hand on screen-right
// Convention B: opposite
// `swap` selects it
float model(bool screenRight, bool swap, float conf = 0.9f) {
  const bool high = screenRight != swap;

  return high ? conf : 1.f - conf;
}

void train(HandednessResolver &r, bool swap, int frames = 40) {
  for (int i = 0; i < frames; ++i) {
    r.learn(0.25f, model(false, swap), 0.75f, model(true, swap));
  }
}
} // namespace

int main() {
  for (bool swap : {false, true}) {
    HandednessResolver r;

    // Untrained: screen position only
    CHECK(!r.trusted());
    CHECK(r.isLeft(0.3f, model(false, swap)));
    CHECK(!r.isLeft(0.7f, model(true, swap)));

    train(r, swap);
    CHECK(r.trusted());
    CHECK((r.polarity() > 0) == !swap);

    // Lone right hand (screen-right type) reaching to far left: still right
    CHECK(!r.isLeft(0.1f, model(true, swap)));
    // Lone left hand reaching over to right half: still left
    CHECK(r.isLeft(0.9f, model(false, swap)));

    // Crossed pair: left hand is on screen-right of right hand
    const float xl = 0.60f, xr = 0.40f;
    const float hl = model(false, swap), hr = model(true, swap);
    CHECK(r.isLeft(xl, hl, &xr, &hr));
    CHECK(!r.isLeft(xr, hr, &xl, &hl));

    // Unsure model output (near 0.5): falls back to screen position
    CHECK(r.isLeft(0.2f, 0.52f));
    CHECK(!r.isLeft(0.8f, 0.48f));

    // Model says same hand twice: no information, screen order wins
    const float ox = 0.7f, oh = model(true, swap);
    CHECK(r.isLeft(0.3f, model(true, swap), &ox, &oh));
  }

  // Learning ignores close hands and ambiguous votes
  {
    HandednessResolver r;
    for (int i = 0; i < 100; ++i) {
      r.learn(0.45f, 0.9f, 0.55f, 0.1f); // too close together
      r.learn(0.2f, 0.52f, 0.8f, 0.48f); // unsure
      r.learn(0.2f, 0.9f, 0.8f, 0.9f);   // model thinks both are same hand
    }
    CHECK(!r.trusted());
  }

  // A few contradictory frames do not flip a trained resolver
  {
    HandednessResolver r;
    train(r, false, 100);
    for (int i = 0; i < 3; ++i) {
      r.learn(0.25f, 0.9f, 0.75f, 0.1f); // flipped
    }
    CHECK(r.trusted() && r.polarity() > 0);
  }

  TEST_MAIN_END();
}
