#!/bin/bash
# Reinstall PyTorch with CPU-only version to avoid CUDA crashes in WSL2

echo "============================================================"
echo "Reinstalling PyTorch with CPU-only version"
echo "============================================================"
echo ""
echo "This will:"
echo "  1. Uninstall current PyTorch (which includes CUDA libraries)"
echo "  2. Install CPU-only version from PyTorch official repository"
echo ""
echo "This is necessary because the WSL2 CUDA driver has a bug that"
echo "causes crashes when PyTorch tries to enumerate CUDA devices."
echo ""
echo "============================================================"
echo ""

# Activate virtualenv
source ~/.virtualenvs/openai/bin/activate

# Check Python version
echo "Python version:"
python --version
echo ""

# Show current PyTorch info
echo "Current PyTorch installation:"
pip show torch | grep -E "Name|Version|Location"
echo ""

# Ask for confirmation
read -p "Do you want to proceed with reinstallation? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Installation cancelled."
    exit 0
fi

echo ""
echo "Step 1: Uninstalling current PyTorch..."
pip uninstall -y torch torchvision torchaudio

echo ""
echo "Step 2: Installing CPU-only PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo ""
echo "============================================================"
echo "Installation complete!"
echo "============================================================"
echo ""

# Verify installation
echo "New PyTorch installation:"
pip show torch | grep -E "Name|Version|Location"
echo ""

echo "Verifying CPU-only mode:"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.device(\"cpu\")}')"

echo ""
echo "============================================================"
echo "You can now run: ./run_trpo_torch.sh --task train --num_timesteps 5000000 --seed 0"
echo "============================================================"
