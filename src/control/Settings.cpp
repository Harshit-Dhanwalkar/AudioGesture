#include "audiogesture/control/Settings.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace {
std::string trim(const std::string &s) {
  const auto a = s.find_first_not_of(" \t\r\n");
  if (a == std::string::npos) {
    return "";
  }

  return s.substr(a, s.find_last_not_of(" \t\r\n") - a + 1);
}

bool parseBool(const std::string &v, bool &out) {
  if (v == "1" || v == "true" || v == "on") {
    out = true;
    return true;
  }

  if (v == "0" || v == "false" || v == "off") {
    out = false;
    return true;
  }

  return false;
}

bool parseFloat(const std::string &v, float &out) {
  char *end = nullptr;
  const float f = std::strtof(v.c_str(), &end);
  if (end == v.c_str() || *end != '\0' || !(f == f) || f > 1e6f || f < -1e6f) {
    return false;
  }
  out = f;

  return true;
}

bool parseInt(const std::string &v, int &out) {
  char *end = nullptr;
  const long n = std::strtol(v.c_str(), &end, 10);
  if (end == v.c_str() || *end != '\0') {
    return false;
  }
  out = static_cast<int>(std::clamp(n, -1000L, 1000L));

  return true;
}
} // namespace

void Settings::clamp() {
  restLength = std::clamp(restLength, 0.15f, 1.2f);
  refHeight = std::clamp(refHeight, 0.08f, 0.8f);
  segSec = std::clamp(segSec, 1.f, 8.f);
  boostMode = std::clamp(boostMode, 0, 2);
  gridStyle = std::clamp(gridStyle, 0, 1);
}

std::string Settings::defaultPath() {
  namespace fs = std::filesystem;
  if (const char *x = std::getenv("XDG_CONFIG_HOME"); x && *x) {
    return (fs::path(x) / "audiogesture" / "settings.ini").string();
  }

  if (const char *h = std::getenv("HOME"); h && *h) {
    return (fs::path(h) / ".config" / "audiogesture" / "settings.ini").string();
  }

  return "audiogesture-settings.ini";
}

bool Settings::load(const std::string &path) {
  std::ifstream in(path);
  if (!in) {
    return false;
  }

  std::string line;
  while (std::getline(in, line)) {
    const auto hash = line.find('#');
    if (hash != std::string::npos) {
      line.erase(hash);
    }

    const auto eq = line.find('=');
    if (eq == std::string::npos) {
      continue;
    }

    const std::string k = trim(line.substr(0, eq)), v = trim(line.substr(eq + 1));
    if (k == "rest_length") {
      parseFloat(v, restLength);
    } else if (k == "ref_height") {
      parseFloat(v, refHeight);
    } else if (k == "segment_seconds") {
      parseFloat(v, segSec);
    } else if (k == "pitch") {
      parseBool(v, pitchOn);
    } else if (k == "temporal_denoise") {
      parseBool(v, temporalDenoise);
    } else if (k == "low_light_boost") {
      parseInt(v, boostMode);
    } else if (k == "grid_style") {
      parseInt(v, gridStyle);
    }
  }

  clamp();

  return true;
}

bool Settings::save(const std::string &path) const {
  namespace fs = std::filesystem;
  std::error_code ec;
  const fs::path p(path);
  if (p.has_parent_path()) {
    fs::create_directories(p.parent_path(), ec);
  }
  std::ofstream out(path, std::ios::trunc);
  if (!out) {
    return false;
  }

  out << "# AudioGesture settings (written on exit; safe to edit)\n"
      << "rest_length = " << restLength << "   # hand distance, in frame heights, for 1x speed\n"
      << "ref_height = " << refHeight << "   # hand height, in frame heights, for full volume\n"
      << "segment_seconds = " << segSec << "\n"
      << "pitch = " << (pitchOn ? "on" : "off") << "\n"
      << "temporal_denoise = " << (temporalDenoise ? "on" : "off") << "\n"
      << "low_light_boost = " << boostMode << "   # 0 auto, 1 on, 2 off\n"
      << "grid_style = " << gridStyle << "   # 0 bars, 1 mirrored\n";

  return static_cast<bool>(out);
}
