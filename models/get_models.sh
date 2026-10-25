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
# Single source of truth for the interpreter. Override with:
#   PYTHON=.venv/bin/python bash models/get_models.sh
PYTHON="${PYTHON:-python3}"
if ! PYTHON=$(command -v "$PYTHON" 2>/dev/null); then
    echo "ERROR: interpreter '$PYTHON' not found; set PYTHON=/path/to/python and re-run" >&2
    exit 1
fi

"$PYTHON" -c 'import sys; print("using:", sys.executable)'

if ONNX_ERR=$("$PYTHON" -c 'import onnx' 2>&1); then
    # onnx available
    [ -f make_opencv_compat.py ] || { echo "ERROR: make_opencv_compat.py missing next to get_models.sh" >&2; exit 1; }
    [ -f "$HAND" ]              || { echo "ERROR: $HAND missing; cannot build $COMPAT" >&2; exit 1; }
    "$PYTHON" make_opencv_compat.py "$HAND" "$COMPAT"
    [ -s "$COMPAT" ] || { echo "ERROR: $COMPAT was not produced" >&2; exit 1; }
    echo "ok      $COMPAT"
else
    # onnx missing
    PYINFO=$("$PYTHON" -c 'import sys; print(sys.version.split()[0], sys.executable)' 2>/dev/null || echo "unknown")
    {
        echo "note: '$PYTHON' cannot import onnx ($PYINFO)."
        [ -n "$ONNX_ERR" ] && printf '%s\n' "$ONNX_ERR" | sed 's/^/      /'
        echo "      Its pip:  $("$PYTHON" -m pip -V 2>&1 || echo 'n/a')"
        echo "      Fix:      \"$PYTHON\" -m pip install onnx   then re-run"
    } >&2

    if [ "$NEED_COMPAT" -eq 1 ]; then
        echo "ERROR: OpenCV $OPENCV_VER (< 4.7) needs $COMPAT, but onnx is missing." >&2
        exit 1
    fi
    echo "      Skipping $COMPAT (OpenCV ${OPENCV_VER:-?} >= 4.7)." >&2
fi
