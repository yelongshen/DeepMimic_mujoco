#!/bin/bash
# Record video comparing PD actions vs refined actions

cd "$(dirname "$0")"

echo "=========================================="
echo "Recording TWO videos for comparison:"
echo "1. Standard PD actions"
echo "2. Refined actions (3 iterations)"
echo "=========================================="
echo ""

# Video 1: Standard PD actions
echo "Recording standard PD actions..."
xvfb-run -a -s "-screen 0 1400x900x24" python record_pd_actions.py \
    --mocap deepmimic_mujoco/motions/humanoid3d_dance_a.txt \
    --output render/pd_standard.avi \
    --duration 30.0 \
    --kp 1.0 \
    --kd 0.1

echo ""
echo "✓ Standard PD video saved"
echo ""

# Video 2: Refined actions
echo "Recording refined actions..."
xvfb-run -a -s "-screen 0 1400x900x24" python record_pd_actions.py \
    --mocap deepmimic_mujoco/motions/humanoid3d_dance_a.txt \
    --output render/pd_refined.avi \
    --duration 30.0 \
    --kp 1.0 \
    --kd 0.1 \
    --refined \
    --refine_iterations 3

echo ""
echo "✓ Refined actions video saved"
echo ""

echo "=========================================="
echo "Comparison videos saved:"
echo "  Standard: render/pd_standard.avi"
echo "  Refined:  render/pd_refined.avi"
echo "=========================================="
echo ""
echo "Watch both to see the improvement from refinement!"
