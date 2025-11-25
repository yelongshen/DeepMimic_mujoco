#!/bin/bash
# Record video of PD control actions

cd "$(dirname "$0")"

# Default: Record PD actions for 60 seconds
echo "Recording PD control actions video..."
xvfb-run -a -s "-screen 0 1400x900x24" python record_pd_actions.py \
    --mocap deepmimic_mujoco/motions/humanoid3d_dance_a.txt \
    --output render/pd_actions_video.avi \
    --duration 60.0 \
    --kp 1.0 \
    --kd 0.1

echo ""
echo "=========================================="
echo "Video saved to: render/pd_actions_video.avi"
echo "=========================================="
