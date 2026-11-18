// Trace record/replay and mapper's invariants on recorded and fuzzed hand motion

#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>

#include "audiogesture/control/GestureMapper.h"
#include "audiogesture/vision/HandTrace.h"
#include "check.h"

namespace {
Hand skeleton(float cx, float wristY, float tipY, float pinch = 1.4f, float open = 0.7f,
              float aspect = 4.f / 3.f) {
  Hand h;
  h.valid = h.hasSkeleton = true;
  h.x = cx;
  h.y = 0.5f * (wristY + tipY);
  h.score = 0.95f;
  for (auto &p : h.lm) {
    p = {cx, h.y};
  }
  const float y9 = wristY + 0.4f * (tipY - wristY), palm = 0.4f * std::fabs(wristY - tipY);
  h.lm[0] = {cx, wristY};
  h.lm[12] = {cx, tipY};
  h.lm[9] = {cx, y9};
  h.lm[5] = {cx - 0.5f * open * palm / aspect, y9};
  h.lm[17] = {cx + 0.5f * open * palm / aspect, y9};
  h.lm[4] = {cx - 0.5f * pinch * palm / aspect, y9 - 0.01f};
  h.lm[8] = {cx + 0.5f * pinch * palm / aspect, y9 - 0.01f};
  return h;
}

bool same(const Hand &a, const Hand &b, float tol) {
  if (a.valid != b.valid)
    return false;
  if (!a.valid)
    return true;
  if (std::fabs(a.x - b.x) > tol || std::fabs(a.y - b.y) > tol || a.hasSkeleton != b.hasSkeleton)
    return false;
  if (a.hasSkeleton)
    for (size_t i = 0; i < a.lm.size(); ++i)
      if (std::fabs(a.lm[i].x - b.lm[i].x) > tol || std::fabs(a.lm[i].y - b.lm[i].y) > tol)
        return false;
  return true;
}

// Run frames through a mapper, checking invariants every step
// Returns last params
EngineParams runChecked(const std::vector<TraceFrame> &fr, const char *what, GestureMapper &m) {
  double prev = fr.empty() ? 0 : fr.front().t;
  EngineParams p;
  for (const TraceFrame &f : fr) {
    const double dt = std::max(1e-3, f.t - prev);
    prev = f.t;
    p = m.update(f.state, dt);
    const bool ok = std::isfinite(p.stretch) && p.stretch >= 0.249f && p.stretch <= 4.001f &&
                    std::isfinite(p.envL) && std::isfinite(p.envR) && std::fabs(p.envL) <= 1.001f &&
                    std::fabs(p.envR) <= 1.001f && std::isfinite(p.pitchSemis) &&
                    std::fabs(p.pitchSemis) <= 12.01f && p.lpHz >= 39.f && p.lpHz <= 20001.f &&
                    p.hpHz >= 9.f && p.hpHz <= 8001.f && std::fabs(p.direction) <= 1.001f;
    if (!ok) {
      std::fprintf(stderr,
                   "invariant broken in %s at t=%.3f: stretch %g envL %g envR %g pitch %g lp %g hp "
                   "%g dir %g\n",
                   what, f.t, p.stretch, p.envL, p.envR, p.pitchSemis, p.lpHz, p.hpHz, p.direction);
      CHECK(ok);
      break;
    }
  }
  return p;
}
} // namespace

int main() {
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path() / "audiogesture_trace_test";
  fs::remove_all(dir);
  fs::create_directories(dir);
  const std::string path = (dir / "a.trace").string();

  // Round trip: full skeletons, a hand without one, an absent hand, odd aspect
  std::vector<TraceFrame> src;
  for (int i = 0; i < 30; ++i) {
    TraceFrame f;
    f.t = i / 30.0;
    f.state.aspect = 16.f / 9.f;
    f.state.left = skeleton(0.2f + 0.01f * i, 0.9f, 0.45f, 1.4f, 0.7f, f.state.aspect);
    if (i % 5 == 0) {
      f.state.right = Hand{}; // dropout
    } else if (i % 5 == 1) {
      f.state.right.valid = true; // skin-colour tracker: position only
      f.state.right.x = 0.7f;
      f.state.right.y = 0.4f;
    } else {
      f.state.right = skeleton(0.8f, 0.9f, 0.5f, 1.4f, 0.7f, f.state.aspect);
    }
    src.push_back(f);
  }
  {
    TraceWriter w;
    CHECK(!w.write(0, src[0].state)); // not open
    CHECK(w.open(path));
    for (const auto &f : src) {
      CHECK(w.write(f.t, f.state));
    }
    CHECK(!w.write(2.0, src.back().state)); // duplicate of previous frame: skipped
    CHECK(w.frames() == src.size());
  }
  std::vector<TraceFrame> got;
  CHECK(loadTrace(path, got));
  CHECK(got.size() == src.size());
  for (size_t i = 0; i < std::min(got.size(), src.size()); ++i) {
    CHECK_NEAR(got[i].t, src[i].t, 1e-5);
    CHECK_NEAR(got[i].state.aspect, src[i].state.aspect, 1e-5);
    CHECK(same(got[i].state.left, src[i].state.left, 1e-5f));
    CHECK(same(got[i].state.right, src[i].state.right, 1e-5f));
  }
  CHECK(!loadTrace((dir / "missing.trace").string(), got));

  // Malformed lines, backwards time and NaNs are skipped, rest survives
  {
    std::ofstream o(dir / "bad.trace");
    o << "# audiogesture-trace v1\n"
         "0.0 1.33 0 0\n"
         "garbage line\n"
         "0.1 1.33 1 0.5 0.5 0.9 0\n"       // Truncated second hand
         "0.2 1.33 1 nan 0.5 0.9 0 0\n"     // NaN
         "0.3 1.33 1 0.5 0.5 0.9 1 1 2 3\n" // Skeleton cut short
         "0.4 1.33 1 0.3 0.6 0.9 0 0\n"
         "0.2 1.33 0 0\n" // Time runs backwards
         "0.5 0.0 0 0\n"; // Bad aspect
  }
  CHECK(loadTrace((dir / "bad.trace").string(), got));
  CHECK(got.size() == 2);

  // Replay: in order, then ends; loop keeps going with increasing time; empty trace ends at once
  {
    ReplaySource r(src, /*realtime=*/false, /*loop=*/false);
    HandState hs;
    size_t n = 0;
    while (r.poll(hs)) {
      CHECK(same(hs.left, src[n].state.left, 0));
      ++n;
    }
    CHECK(n == src.size());
    CHECK(!r.frame().empty());

    ReplaySource lp(src, false, true);
    for (size_t i = 0; i < 3 * src.size(); ++i) {
      CHECK(lp.poll(hs));
    }
    ReplaySource none({}, false, true);
    CHECK(!none.poll(hs));
  }

  // Real-time pacing honours timestamps (0.3 s of trace takes about 0.3 s)
  {
    std::vector<TraceFrame> s3(src.begin(), src.begin() + 10); // t = 0 .. 0.3
    ReplaySource r(s3, true, false);
    HandState hs;
    const auto t0 = std::chrono::steady_clock::now();
    while (r.poll(hs)) {
    }
    const double took =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    CHECK(took > 0.25 && took < 0.6);
  }

  // Scenario: hands start 0.5 apart and are brought together over 2 s -> speed rises, stays in range
  {
    std::vector<TraceFrame> fr;
    for (int i = 0; i <= 60; ++i) {
      TraceFrame f;
      f.t = i / 30.0;
      const float gap = 0.30f - 0.22f * i / 60.f;
      f.state.left = skeleton(0.5f - gap, 0.9f, 0.45f);
      f.state.right = skeleton(0.5f + gap, 0.9f, 0.45f);
      fr.push_back(f);
    }
    GestureMapper m;
    float firstStretch = 0, lastStretch = 0;
    double prev = 0;
    for (size_t i = 0; i < fr.size(); ++i) {
      const EngineParams &p = m.update(fr[i].state, std::max(1e-3, fr[i].t - prev));
      prev = fr[i].t;
      if (i == 5)
        firstStretch = p.stretch;
      lastStretch = p.stretch;
    }
    CHECK(firstStretch > 2.f * lastStretch);
    CHECK(firstStretch <= 4.001f && lastStretch >= 0.249f);
    CHECK(!m.frozen());
  }

  // Fuzz: jittery, dropping out, pinching and rolling; invariants hold every frame and
  // nothing stays frozen/filtered once hands have been gone for a while
  for (unsigned seed = 1; seed <= 20; ++seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(0.f, 1.f);
    std::vector<TraceFrame> fr;
    int outL = 0, outR = 0;
    for (int i = 0; i < 400; ++i) {
      TraceFrame f;
      f.t = i / 30.0 + 0.001 * (seed % 3);
      if (outL == 0 && u(rng) < 0.03f)
        outL = 1 + int(u(rng) * 20);
      if (outR == 0 && u(rng) < 0.03f)
        outR = 1 + int(u(rng) * 20);
      auto pose = [&](float side) {
        const float tip =
            0.2f + 0.7f * u(rng) - 0.1f; // sometimes very short, sometimes upside down
        return skeleton(std::clamp(side + 0.3f * (u(rng) - 0.5f), 0.f, 1.f), 0.9f * u(rng) + 0.05f,
                        tip, 2.f * u(rng), 0.05f + 1.0f * u(rng));
      };
      if (outL > 0)
        --outL;
      else
        f.state.left = pose(0.25f);
      if (outR > 0)
        --outR;
      else
        f.state.right = pose(0.75f);
      fr.push_back(f);
    }
    for (int i = 0; i < 40; ++i) { // hands leave
      TraceFrame f;
      f.t = 400 / 30.0 + i / 30.0;
      fr.push_back(f);
    }
    GestureMapper m;
    const EngineParams last = runChecked(fr, "fuzz", m);
    CHECK(!last.freeze);
    CHECK(last.lpHz > 15000.f && last.hpHz < 40.f);
  }

  // Every trace dropped into tests/data/
  int recorded = 0;
  const fs::path data = fs::path(AG_SOURCE_DIR) / "tests" / "data";
  if (fs::exists(data)) {
    for (const auto &e : fs::directory_iterator(data)) {
      if (e.path().extension() != ".trace")
        continue;
      std::vector<TraceFrame> fr;
      CHECK(loadTrace(e.path().string(), fr));
      CHECK(!fr.empty());
      GestureMapper m;
      runChecked(fr, e.path().filename().string().c_str(), m);
      ++recorded;
    }
  }
  std::printf("(%d recorded trace(s) checked)\n", recorded);

  fs::remove_all(dir);
  TEST_MAIN_END();
}
