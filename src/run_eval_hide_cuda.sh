#!/bin/bash
# Run evaluation with CUDA driver completely hidden from TensorFlow

# Save original library path
ORIGINAL_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"

# Remove all CUDA-related paths
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v cuda | grep -v wsl | tr '\n' ':')

# Set environment to force CPU
export CUDA_VISIBLE_DEVICES=""
export TF_CPP_MIN_LOG_LEVEL="2"

# Force GLFW rendering for Xvfb (will be run with xvfb-run)
export MUJOCO_GL="glfw"

echo "Running with CPU only - CUDA libraries hidden from search path"
echo "Using GLFW rendering with Xvfb virtual display"
echo "Modified LD_LIBRARY_PATH to exclude CUDA directories"

cd /mnt/c/DeepMimic_mujoco/src

# Check if xvfb-run is available
if command -v xvfb-run &> /dev/null; then
    echo "Running with xvfb-run for virtual display..."
    xvfb-run -a -s "-screen 0 1024x768x24" python trpo.py --task evaluate --load_model_path checkpoint_tmp/DeepMimic/trpo-walk-0/DeepMimic/trpo-walk-0 "$@"
else
    echo "ERROR: xvfb-run not found. Please install it:"
    echo "  sudo apt-get install xvfb"
    exit 1
fi

# Restore library path
export LD_LIBRARY_PATH="$ORIGINAL_LD_LIBRARY_PATH"
