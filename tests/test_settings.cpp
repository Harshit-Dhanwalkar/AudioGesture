// Settings
#include <filesystem>
#include <fstream>

#include "audiogesture/control/Settings.h"
#include "check.h"

int main() {
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path() / "audiogesture_settings_test";
  fs::remove_all(dir);
  const std::string path = (dir / "sub" / "settings.ini").string();

  Settings missing;
  CHECK(!missing.load(path)); // no file: false, defaults kept
  CHECK_NEAR(missing.restLength, 0.45, 1e-6);

  Settings a;
  a.restLength = 0.62f;
  a.refHeight = 0.33f;
  a.segSec = 2.f;
  a.pitchOn = true;
  a.temporalDenoise = false;
  a.boostMode = 2;
  a.gridStyle = 1;
  CHECK(a.save(path)); // creates directory too

  Settings b;
  CHECK(b.load(path));
  CHECK_NEAR(b.restLength, 0.62, 1e-4);
  CHECK_NEAR(b.refHeight, 0.33, 1e-4);
  CHECK_NEAR(b.segSec, 2.0, 1e-6);
  CHECK(b.pitchOn && !b.temporalDenoise && b.boostMode == 2 && b.gridStyle == 1);

  // Hand-edited nonsense: ignored or clamped, never fatal
  {
    std::ofstream o(path, std::ios::trunc);
    o << "rest_length = 99\nref_height = banana\nsegment_seconds=-5\nlow_light_boost = 7\n"
         "this line has no equals\nunknown = 3\npitch = maybe\n= \n# comment only\n";
  }
  Settings c;
  CHECK(c.load(path));
  CHECK_NEAR(c.restLength, 1.2, 1e-6); // clamped to maximum
  CHECK_NEAR(c.refHeight, 0.45, 1e-6); // unparsable: default kept
  CHECK_NEAR(c.segSec, 1.0, 1e-6);     // clamped to minimum
  CHECK(c.boostMode == 2);             // clamped
  CHECK(!c.pitchOn);                   // unparsable: default kept

  CHECK(!Settings::defaultPath().empty());
  fs::remove_all(dir);
  TEST_MAIN_END();
}
