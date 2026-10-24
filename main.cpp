#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "AudioEngine.h"
#include "GestureMapper.h"
#include "GestureTracker.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <audio.wav|flac|ogg|mp3> [camera_index]\n", argv[0]);
        return 1;
    }
    AudioEngine audio;
    if (!audio.load(argv[1])) return 1;

    SkinHandTracker tracker(argc > 2 ? std::atoi(argv[2]) : 0);
    if (!tracker.isOpen()) { std::fprintf(stderr, "cannot open camera\n"); return 1; }
    if (!audio.start())    { std::fprintf(stderr, "cannot start audio\n"); return 1; }

    GestureMapper mapper;
    HandState hands;
    auto last = std::chrono::steady_clock::now();

    while (tracker.poll(hands)) {
        auto now = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(now - last).count();
        last = now;

        mapper.update(hands, dt, audio);

        cv::Mat view = tracker.frame().clone();
        mapper.draw(view, hands, audio.playheadNorm());
        cv::imshow("gesture_grain", view);
        int k = cv::waitKey(1);
        if (k == 27 || k == 'q') break;
    }
    audio.stop();
    return 0;
}
