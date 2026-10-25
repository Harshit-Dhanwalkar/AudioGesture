# gesture_grain

Control audio with two hands and a webcam: granular time-stretch, pitch, reverse/scrub -
with a real 21-joint hand skeleton tracker that keeps working in dim light, and a cava-style
spectrum visualizer of what you are actually hearing.

## Setup

```sh
sudo apt install build-essential cmake pkg-config libopencv-dev portaudio19-dev libsndfile1-dev
pip install onnx        # only needed once, for `make models`, if OpenCV <=  4.7
```

## Build & run

```sh
# export PATH="/usr/bin:$PATH:/home/linuxbrew/.linuxbrew/bin"
make models                       # one-off: downloads ~8 MB of hand-tracking models into models/
make                              # configure + build
make run AUDIO=sample.wav         # or:  ./build/gesture_grain sample.wav [camera_index | video_file]
```

If `models/` is empty the app says so and falls back to the old skin-colour tracker (`--skin` forces it).
`make clean`, `make distclean`, `make rebuild`, `make help` as before.

MP3 needs libsndfile >= 1.1; otherwise convert to WAV first.

## Gestures

| Gesture                            | Parameter                                       |
| ---------------------------------- | ----------------------------------------------- |
| Distance between hands (palm centres) | Time stretch 0.25x-4x and grain size 40-250 ms  |
| Right hand height                  | Pitch, +12 to -12 semitones                     |
| Fast right-hand sweep left / right | Latch reverse / forward (smoothed through zero) |

With one hand in view it is treated as the right hand (pitch + direction). When a second hand appears the one further left on screen becomes the left hand, and each hand keeps its role after that.

## Window and keys

Camera view with the skeleton and live parameters on top, the **spectrum** under it (log-frequency bars, auto-sensitivity, gravity fall-off, peak caps; computed from the audio *after* stretch/pitch/reverse) and a **waveform overview** of the file with the playhead at the bottom.

| Key      | Action                                                         |
| -------- | -------------------------------------------------------------- |
| `l`      | low-light boost: auto -> on -> off                             |
| `v`      | spectrum style: bars <-> mirrored                              |
| `d`      | show the tracker's search windows (debug)                      |
| `q`/Esc  | quit                                                           |

## How the tracker works

Two small neural nets from MediaPipe, run with OpenCV's `cv::dnn` (no extra runtime):

1. **Palm detector** (192x192) finds palms anywhere in the frame.
2. **Hand landmark model** (224x224) regresses 21 joints inside a rotated crop around one hand, and says how
   confident it is that a hand is really there.

The joints from frame N give the crop for frame N+1, so palm detector only runs when a hand is missing. Joints are smoothed with a One-Euro filter (little lag when you move fast, no jitter when you hold still). The control point is the centre of wrist + four finger bases, which doesn't move when you curl your fingers.

### Low light

* The `LowLightEnhancer` switches on by itself when the frame's mean brightness drops below ~85/255 (and off again above ~105): a gamma lift towards mid-grey plus a blur that grows as the scene gets darker, to average away the amplified sensor noise. The preview shows the brightened frame, so you see what the tracker sees.
* Palm candidates are accepted at a low score (0.3) and verified by the landmark model's own confidence (0.5).
* In synthetic tests (a real photo darkened to 7-25% brightness with sensor noise) this found a hand where the un-enhanced frame found nothing below ~25%. Very strong noise at very low light still defeats it:
  **light the room a little, or from the front** — a lamp or even the monitor helps far more than software.
* Webcams drop their frame rate in the dark to lengthen exposure. See what your camera offers with `v4l2-ctl -d /dev/video0 --list-ctrls`; `exposure_dynamic_framerate=0` keeps 30 fps (darker but smoother) and raising `gain`/`brightness` trades noise for light.

### Tuning

| What                              | Where                                                   |
| --------------------------------- | ------------------------------------------------------- |
| Detection / presence thresholds, smoothing | `HandTrackerConfig` in `HandTracker.h`          |
| Low-light on/off levels, blur, gamma target | `LowLightEnhancer.h`                           |
| Audio parameter smoothing         | `AudioEngine::start`                                    |
| Hand coordinate smoothing, gesture ranges | `GestureMapper::update`                         |
| Bars, range, gravity, monstercat  | `Visualizer::Config` in `Visualizer.h`                  |

### Testing the tracker without a camera

```sh
make probe
build/hand_probe photo.jpg --out result.png                       # skeleton on a still image
build/hand_probe photo.jpg --dark 0.15 --noise 4 --lowlight --out dark.png   # simulate a dim room
build/hand_probe clip.mp4 --frames 120 --debug --out last.png     # a video file; prints per-frame timings
./build/gesture_grain sample.wav clip.mp4                         # whole app on a recorded clip (loops)
```

## OpenCV versions

Works with OpenCV >= 4.5 (tested on 4.6). The stock landmark model uses ONNX operators that OpenCV < 4.7
can't parse; `make models` therefore also writes `handpose_estimation_mediapipe_2023feb_cvcompat.onnx`
(needs `pip install onnx`) via `models/make_opencv_compat.py`, and the app picks the right file by itself.
The rewrite is numerically identical (checked against ONNX Runtime: max difference ~5e-5 px).

## Files

```
main.cpp                 window, keys, wiring
AudioEngine.h/.cpp       granular playback engine + lock-free output tap for the visualizer
GestureMapper.h          hand state -> audio parameters (the maths is commented here)
GestureTracker.h/.cpp    HandState types, IHandSource interface, old skin-colour tracker (fallback)
HandTracker.h/.cpp       MediaPipe palm + landmark tracker, camera source, skeleton drawing
LowLightEnhancer.h       dark-frame boost         OneEuroFilter.h   jitter filter
Visualizer.h/.cpp        FFT spectrum bars + waveform overview
tools/hand_probe.cpp     camera-less tracker test
models/                  get_models.sh, make_opencv_compat.py  (+ the .onnx files after `make models`)
```

Models: MediaPipe Hands, Apache-2.0, as packaged by [opencv_zoo](https://github.com/opencv/opencv_zoo).
