# Audio Gesture

Control audio with two hands and a webcam: time-stretch, pitch, reverse/scrub.

## Setup

```sh
sudo apt install build-essential cmake pkg-config libopencv-dev portaudio19-dev libsndfile1-dev
```

## Build & run

```sh
export PATH="/usr/bin:$PATH:/home/linuxbrew/.linuxbrew/bin"
cmake -S . -B build && cmake --build build -j
./build/gesture_grain sample.wav [camera_index]
```

Esc or q quits. MP3 needs libsndfile >= 1.1; otherwise convert to WAV first.

## Gestures

| Gesture                            | Parameter                                       |
| ---------------------------------- | ----------------------------------------------- |
| Distance between hands             | Time stretch 0.25x-4x and grain size 40-250 ms  |
| Right hand height                  | Pitch, +12 to -12 semitones                     |
| Fast right-hand sweep left / right | Latch reverse / forward (smoothed through zero) |

