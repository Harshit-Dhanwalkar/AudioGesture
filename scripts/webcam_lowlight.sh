#!/usr/bin/env bash
# Get more light out of a UVC webcam *before* any software enhancement.
# Cameras in auto mode lower their frame rate in the dark to lengthen the
# exposure, which smears moving hands; and many default to low gain.
#
#   scripts/webcam_lowlight.sh            # /dev/video0, show what is available
#   scripts/webcam_lowlight.sh /dev/video2 apply
#
# Needs v4l-utils:  sudo apt install v4l-utils
set -euo pipefail
DEV=${1:-/dev/video0}
MODE=${2:-show}

command -v v4l2-ctl >/dev/null || { echo "install v4l-utils first"; exit 1; }

if [ "$MODE" = show ]; then
    v4l2-ctl -d "$DEV" --list-ctrls
    echo
    echo "Re-run with:  $0 $DEV apply"
    exit 0
fi

set_ctrl() { v4l2-ctl -d "$DEV" --set-ctrl="$1" 2>/dev/null && echo "set $1" || echo "skip $1 (not supported)"; }

# Keep 30 fps instead of letting auto-exposure drop to 5-10 fps in the dark
set_ctrl exposure_dynamic_framerate=0
# Aperture-priority auto exposure (3) with the longest exposure the frame rate allows
set_ctrl auto_exposure=3
# More analogue gain is much cleaner than software gain; raise it a bit at a time
set_ctrl gain=64
set_ctrl brightness=140
# Backlight compensation helps when a monitor is behind you, hurts otherwise
set_ctrl backlight_compensation=0
echo "done. If the picture is still dark, light the room from the FRONT (a lamp or the monitor)."
