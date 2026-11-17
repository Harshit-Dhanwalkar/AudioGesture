cmake_minimum_required(VERSION 3.16)
project(AudioGesture LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(AUDIOGESTURE_BUILD_TESTS "Build the unit tests (ctest)" ON)
option(AUDIOGESTURE_WERROR "Treat warnings as errors (CI)" OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui videoio objdetect dnn)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PORTAUDIO REQUIRED IMPORTED_TARGET portaudio-2.0)
pkg_check_modules(SNDFILE   REQUIRED IMPORTED_TARGET sndfile)

add_library(audiogesture_core STATIC
  src/audio/AudioEngine.cpp
  src/audio/Recorder.cpp
  src/audio/SpectrumAnalyzer.cpp
  src/control/GestureMapper.cpp
  src/control/Settings.cpp
  src/ui/SpectrumGrid.cpp
  src/vision/HandTracker.cpp
  src/vision/SkinHandTracker.cpp)
target_include_directories(audiogesture_core PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(audiogesture_core PUBLIC
  ${OpenCV_LIBS} PkgConfig::PORTAUDIO PkgConfig::SNDFILE)
target_compile_options(audiogesture_core PRIVATE -Wall -Wextra -O2)
if(AUDIOGESTURE_WERROR)
  target_compile_options(audiogesture_core PRIVATE -Werror)
endif()

# App
add_executable(audiogesture src/main.cpp)
target_link_libraries(audiogesture PRIVATE audiogesture_core)
target_compile_options(audiogesture PRIVATE -Wall -Wextra -O2)

# Camera-less tracker test: `make probe`, then  build/hand_probe photo.jpg --out result.png
add_executable(hand_probe EXCLUDE_FROM_ALL tools/hand_probe.cpp)
target_link_libraries(hand_probe PRIVATE audiogesture_core)

# Tests: `make test`  (= cmake --build build && ctest --test-dir build)
if(AUDIOGESTURE_BUILD_TESTS)
  enable_testing()
  set(AG_TESTS test_audio_engine test_recorder test_settings test_mapper test_spectrum test_enhancer test_grid test_tracker)
  foreach(t ${AG_TESTS})
    add_executable(${t} tests/${t}.cpp)
    target_link_libraries(${t} PRIVATE audiogesture_core)
    target_include_directories(${t} PRIVATE tests)
    add_test(NAME ${t} COMMAND ${t} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
  endforeach()
  # test_tracker needs the ONNX models (`make models`); it reports SKIP (exit 77)
  # when they are absent, which ctest shows as skipped, not failed.
  set_tests_properties(test_tracker PROPERTIES SKIP_RETURN_CODE 77 LABELS models)
endif()
