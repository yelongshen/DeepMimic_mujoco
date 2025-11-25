#!/bin/bash
# Test that trpo_torch.py is syntactically correct

cd /mnt/c/DeepMimic_mujoco/src

echo "Testing trpo_torch.py syntax..."
python3 -m py_compile trpo_torch.py

if [ $? -eq 0 ]; then
    echo "✓ File syntax is OK!"
    echo ""
    echo "You can now run training:"
    echo "  ./run_trpo_torch.sh --task train --num_timesteps 100000 --seed 0"
else
    echo "✗ File has syntax errors"
    exit 1
fi
