#pragma once
// Pick an audio file with desktop's own dialog when none was given on command line
// Uses `zenity` (GNOME), else `kdialog`; no extra library dependency
// Returns "" if there is no dialog tool or user cancelled

#include <functional>
#include <string>

// run(cmd) executes a shell command and returns its stdout ("" on failure)
// Injectable so tests need no desktop
using CommandRunner = std::function<std::string(const std::string &)>;

std::string systemRunner(const std::string &cmd);
std::string pickAudioFile(const CommandRunner &run = systemRunner);
