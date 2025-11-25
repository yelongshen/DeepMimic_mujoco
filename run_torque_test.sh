#!/bin/bash
# Wrapper script to run torque control test with video recording on headless servers
# This uses xvfb to create a virtual display for rendering

cd "$(dirname "$0")/src"

echo "Starting torque control test with video recording..."
echo "This uses PD control to track mocap reference motion"
echo "Video will be saved to: ./render/"
echo ""

# Check if xvfb is installed
if ! command -v xvfb-run &> /dev/null; then
    echo "ERROR: xvfb-run not found!"
    echo "Install with: sudo apt-get install -y xvfb"
    exit 1
fi

# Run with xvfb virtual display
xvfb-run -a -s "-screen 0 1400x900x24" python env_torque_test.py

echo ""
echo "Done! Check ./src/render/ for the video file."
