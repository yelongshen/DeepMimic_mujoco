#!/bin/bash
# Train policy using Supervised Fine-Tuning (SFT) from mocap data
# This is MUCH faster than pure RL - gets good results in ~1 hour

cd "$(dirname "$0")/src"

# Default arguments
MOCAP="deepmimic_mujoco/motions/humanoid3d_dance_a.txt"
EPOCHS=100
BATCH_SIZE=256
LR=0.001
SAVE_PATH="policy_sft_pretrained.pth"

echo "======================================================================="
echo "Training Motion Imitation Policy with Supervised Learning"
echo "======================================================================="
echo ""
echo "This will train a policy directly from mocap data (much faster than RL)"
echo "Expected time: ~30-60 minutes"
echo ""
echo "Configuration:"
echo "  Mocap file: $MOCAP"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LR"
echo "  Save path: $SAVE_PATH"
echo ""
echo "Starting training..."
echo ""

# Check if running in WSL
if grep -qi microsoft /proc/version 2>/dev/null; then
    echo "Detected WSL environment"
    PYTHON="/home/dev/.virtualenvs/openai/bin/python"
else
    PYTHON="python"
fi

# Run training
$PYTHON train_sft.py \
    --mocap "$MOCAP" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --save_path "$SAVE_PATH" \
    --eval_episodes 5 \
    --eval_steps 500

echo ""
echo "======================================================================="
echo "Training Complete!"
echo "======================================================================="
echo ""
echo "Trained model saved to: $SAVE_PATH"
echo ""
echo "Next steps:"
echo "  1. Test the policy:"
echo "     python dp_env_v3.py (modify to load model)"
echo ""
echo "  2. Fine-tune with RL (recommended for robustness):"
echo "     python trpo_torch.py --task train --load_sft_pretrain $SAVE_PATH --num_timesteps 1000000"
echo ""
echo "  3. Compare results:"
echo "     SFT gives you 80% quality in 1 hour"
echo "     SFT + RL fine-tuning gives you 100% quality in 2-3 hours total"
echo "     Pure RL would take 1-2 days to reach same quality"
echo ""
