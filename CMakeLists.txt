cmake_minimum_required(VERSION 3.16)
project(GestureGrain LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio objdetect)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PORTAUDIO REQUIRED IMPORTED_TARGET portaudio-2.0)
pkg_check_modules(SNDFILE   REQUIRED IMPORTED_TARGET sndfile)

add_executable(gesture_grain
  main.cpp
  AudioEngine.cpp
  GestureTracker.cpp)

target_include_directories(gesture_grain PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(gesture_grain PRIVATE
  ${OpenCV_LIBS} PkgConfig::PORTAUDIO PkgConfig::SNDFILE)
target_compile_options(gesture_grain PRIVATE -Wall -Wextra -O2)
