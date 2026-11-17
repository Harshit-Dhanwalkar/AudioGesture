#include "audiogesture/ui/FilePicker.h"

#include <cstdio>

std::string systemRunner(const std::string &cmd) {
  FILE *p = popen(cmd.c_str(), "r");
  if (!p) {
    return "";
  }

  std::string out;
  char buf[512];
  while (std::fgets(buf, sizeof buf, p)) {
    out += buf;
  }
  pclose(p);

  return out;
}

namespace {
std::string firstLine(const std::string &s) {
  const auto e = s.find_first_of("\r\n");
  return e == std::string::npos ? s : s.substr(0, e);
}
} // namespace

// Only fixed command strings are built here: nothing from environment or user is interpolated into
// a shell command
std::string pickAudioFile(const CommandRunner &run) {
  if (!firstLine(run("command -v zenity 2>/dev/null")).empty()) {
    return firstLine(
        run("zenity --file-selection --title='Choose an audio file' "
            "--file-filter='Audio | *.wav *.flac *.ogg *.mp3 *.WAV *.FLAC *.OGG *.MP3' "
            "--file-filter='All files | *' 2>/dev/null"));
  }

  if (!firstLine(run("command -v kdialog 2>/dev/null")).empty()) {
    return firstLine(run("kdialog --getopenfilename . "
                         "'*.wav *.flac *.ogg *.mp3|Audio files' 2>/dev/null"));
  }

  return "";
}
