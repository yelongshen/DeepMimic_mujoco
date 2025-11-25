#!/bin/bash
# Run TRPO with PyTorch (CPU-only mode)

# CRITICAL: Prevent CUDA entirely
export CUDA_VISIBLE_DEVICES=""
export TF_CPP_MIN_LOG_LEVEL="2"
export MUJOCO_GL="glfw"

echo "========================================================"
echo "Running PyTorch TRPO with CPU-only mode"
echo "========================================================"
echo ""
echo "NOTE: If you see CUDA crashes, you need to reinstall PyTorch"
echo "with CPU-only version using:"
echo "  pip uninstall torch torchvision torchaudio"
echo "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
echo ""
echo "========================================================"

cd /mnt/c/DeepMimic_mujoco/src

# Check if xvfb-run is available
if command -v xvfb-run &> /dev/null; then
    echo "Running with xvfb-run for virtual display..."
    xvfb-run -a -s "-screen 0 1024x768x24" python trpo_torch.py "$@"
else
    echo "ERROR: xvfb-run not found. Please install it:"
    echo "  sudo apt-get install xvfb"
    exit 1
fi
