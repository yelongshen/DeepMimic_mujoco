#!/bin/bash
# Wrapper script to run DeepMimic with video recording on headless servers
# This uses xvfb to create a virtual display for rendering

cd "$(dirname "$0")/src"

echo "Starting DeepMimic with video recording..."
echo "Video will be saved to: ./render/"
echo ""

# Check if xvfb is installed
if ! command -v xvfb-run &> /dev/null; then
    echo "ERROR: xvfb-run not found!"
    echo "Install with: sudo apt-get install -y xvfb"
    exit 1
fi

# Run with xvfb virtual display
xvfb-run -a -s "-screen 0 1400x900x24" python dp_env_v3.py --load_model policy_sft_pretrained.pth

echo ""
echo "Done! Check ./src/render/ for the video file."
