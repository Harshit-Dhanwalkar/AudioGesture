#!/usr/bin/env bash
# Downloads the two MediaPipe ONNX models used by the hand tracker (Apache-2.0, via opencv_zoo)
# and, if the `onnx` Python package is available, makes an OpenCV<4.7-compatible copy of the
# landmark model.  Safe to re-run: existing good files are kept.
set -euo pipefail
cd "$(dirname "$0")"

# Pinned opencv_zoo commit so everyone gets byte-identical models.
COMMIT=48b10fa505aeb51fbd127fc60a5c660040e823b7
PALM=palm_detection_mediapipe_2023feb.onnx
HAND=handpose_estimation_mediapipe_2023feb.onnx
PALM_SHA=78ff51c38496b7fc8b8ebdb6cc8c1abb02fa6c38427c6848254cdaba57fcce7c
HAND_SHA=db0898ae717b76b075d9bf563af315b29562e11f8df5027a1ef07b02bef6d81c

fetch() {  # fetch <dir-in-zoo> <file> <sha256>
    local dir=$1 file=$2 sha=$3
    if [ -s "$file" ] && [ "$(sha256sum "$file" | cut -d' ' -f1)" = "$sha" ]; then
        echo "ok      $file"; return
    fi
    # The files are git-LFS objects: github.com/.../raw redirects to media.githubusercontent.com,
    # which we can also hit directly.
    local urls=(
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/$COMMIT/models/$dir/$file"
        "https://github.com/opencv/opencv_zoo/raw/$COMMIT/models/$dir/$file")
    for u in "${urls[@]}"; do
        echo "fetch   $u"
        if command -v curl >/dev/null; then curl -fsSL --retry 2 -o "$file.tmp" "$u" || continue
        else wget -q -O "$file.tmp" "$u" || continue; fi
        if [ "$(sha256sum "$file.tmp" | cut -d' ' -f1)" = "$sha" ]; then mv "$file.tmp" "$file"; echo "ok      $file"; return; fi
        echo "checksum mismatch for $u" >&2
    done
    rm -f "$file.tmp"
    echo "FAILED to download $file" >&2
    exit 1
}

fetch palm_detection_mediapipe      "$PALM" "$PALM_SHA"
fetch handpose_estimation_mediapipe "$HAND" "$HAND_SHA"

COMPAT=${HAND%.onnx}_cvcompat.onnx
if python3 -c 'import onnx' 2>/dev/null; then
    python3 make_opencv_compat.py "$HAND" "$COMPAT"
else
    echo "note: python package 'onnx' not found, so $COMPAT was not generated." >&2
    echo "      Only needed if your OpenCV is older than 4.7 ('pkg-config --modversion opencv4')." >&2
    echo "      Fix:  pip install onnx   (or: sudo apt install python3-onnx)  and re-run this script." >&2
fi
echo "done."
