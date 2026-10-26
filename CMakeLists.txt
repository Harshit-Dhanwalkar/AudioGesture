cmake_minimum_required(VERSION 3.16)
project(GestureGrain LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui videoio objdetect dnn)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PORTAUDIO REQUIRED IMPORTED_TARGET portaudio-2.0)
pkg_check_modules(SNDFILE   REQUIRED IMPORTED_TARGET sndfile)

# The app
add_executable(audiogesture
  main.cpp
  AudioEngine.cpp
  GestureTracker.cpp
  HandTracker.cpp
  Visualizer.cpp)

target_include_directories(audiogesture PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(audiogesture PRIVATE
  ${OpenCV_LIBS} PkgConfig::PORTAUDIO PkgConfig::SNDFILE)
target_compile_options(audiogesture PRIVATE -Wall -Wextra -O2)

# Camera-less tracker test: `make probe`, then  build/hand_probe photo.jpg --out result.png
add_executable(hand_probe EXCLUDE_FROM_ALL tools/hand_probe.cpp HandTracker.cpp)
target_include_directories(hand_probe PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR})
target_link_libraries(hand_probe PRIVATE ${OpenCV_LIBS})
target_compile_options(hand_probe PRIVATE -Wall -Wextra -O2)
