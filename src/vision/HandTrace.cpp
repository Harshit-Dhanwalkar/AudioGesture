#include "audiogesture/vision/HandTrace.h"

#include <cmath>
#include <iomanip>
#include <sstream>
#include <thread>

namespace {
const char *kHeader = "# audiogesture-trace v1";

void putHand(std::ostream &o, const Hand &h) {
  if (!h.valid) {
    o << " 0";
    return;
  }

  o << " 1 " << h.x << ' ' << h.y << ' ' << h.score << ' ' << int(h.hasSkeleton);
  if (h.hasSkeleton) {
    for (const auto &p : h.lm) {
      o << ' ' << p.x << ' ' << p.y;
    }
  }
}

bool getHand(std::istream &in, Hand &h) {
  int valid = 0;
  if (!(in >> valid) || (valid != 0 && valid != 1)) {
    return false;
  }

  h = Hand{};
  if (!valid) {
    return true;
  }

  int sk = 0;
  if (!(in >> h.x >> h.y >> h.score >> sk) || !std::isfinite(h.x) || !std::isfinite(h.y)) {
    return false;
  }

  h.valid = true;
  h.hasSkeleton = sk != 0;
  if (h.hasSkeleton) {
    for (auto &p : h.lm) {
      if (!(in >> p.x >> p.y) || !std::isfinite(p.x) || !std::isfinite(p.y)) {
        return false;
      }
    }
  }

  return true;
}

bool sameHand(const Hand &a, const Hand &b) {
  if (a.valid != b.valid || (!a.valid)) {
    return a.valid == b.valid;
  }
  if (a.x != b.x || a.y != b.y || a.hasSkeleton != b.hasSkeleton) {
    return false;
  }
  if (a.hasSkeleton) {
    for (size_t i = 0; i < a.lm.size(); ++i) {
      if (a.lm[i] != b.lm[i]) {
        return false;
      }
    }
  }

  return true;
}
} // namespace

bool TraceWriter::open(const std::string &path) {
  out_.open(path, std::ios::trunc);
  if (out_) {
    out_ << kHeader << '\n';
    out_ << std::setprecision(7);
  }

  return static_cast<bool>(out_);
}

bool TraceWriter::write(double t, const HandState &hs) {
  if (!out_.is_open() || !out_) {
    return false;
  }
  if (haveLast_ && sameHand(hs.left, last_.left) && sameHand(hs.right, last_.right)) {
    return false;
  }

  out_ << t << ' ' << hs.aspect;
  putHand(out_, hs.left);
  putHand(out_, hs.right);
  out_ << '\n';
  last_ = hs;
  haveLast_ = true;
  ++n_;

  return true;
}

bool loadTrace(const std::string &path, std::vector<TraceFrame> &out) {
  std::ifstream in(path);
  if (!in) {
    return false;
  }

  out.clear();
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }

    std::istringstream ss(line);
    TraceFrame f;
    if (!(ss >> f.t >> f.state.aspect) || !std::isfinite(f.t) || !(f.state.aspect > 0.1f) ||
        !getHand(ss, f.state.left) || !getHand(ss, f.state.right)) {
      continue; // malformed: skip
    }
    if (!out.empty() && f.t < out.back().t) {
      continue; // time must not run backwards
    }

    out.push_back(f);
  }

  return true;
}

ReplaySource::ReplaySource(std::vector<TraceFrame> frames, bool realtime, bool loop)
    : frames_(std::move(frames)), realtime_(realtime), loop_(loop),
      blank_(480, 640, CV_8UC3, cv::Scalar(40, 40, 40)) {}

bool ReplaySource::poll(HandState &out) {
  if (frames_.empty()) {
    return false;
  }

  if (next_ == 0 && start_.time_since_epoch().count() == 0) {
    start_ = std::chrono::steady_clock::now();
  }
  if (next_ >= frames_.size()) {
    if (!loop_) {
      return false;
    }

    offset_ += frames_.back().t + 0.033;
    next_ = 0;
  }

  const TraceFrame &f = frames_[next_++];
  if (realtime_) {
    const auto due = start_ + std::chrono::microseconds(long((f.t + offset_) * 1e6));
    std::this_thread::sleep_until(due);
  }

  out = f.state;

  return true;
}
