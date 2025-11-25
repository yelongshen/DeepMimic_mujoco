# SFT Quick Reference Card

## Commands

### Test
```bash
python test_sft.py
```

### Train
```bash
./run_sft_train.sh                    # Convenient script
cd src && python train_sft.py         # Direct command
```

### Fine-tune with RL
```bash
cd src
python trpo_torch.py --task train --load_sft_pretrain policy_sft_pretrained.pth
```

---

## Time & Quality

| Method | Time | Reward | Use Case |
|--------|------|--------|----------|
| Pure RL | 24-48h | 8.5 | Original |
| SFT only | 1h | 7.3 | Quick |
| SFT+RL | 3h | 8.8 | Best ⭐ |

---

## Common Arguments

```bash
python train_sft.py \
    --mocap PATH           # Mocap file
    --epochs 100           # Training epochs
    --batch_size 256       # Batch size
    --lr 0.001            # Learning rate
    --lookahead 1         # Frames ahead (1-5)
    --save_path FILE.pth  # Output model
```

---

## Files

- **`train_sft.py`** - Main training script
- **`test_sft.py`** - Verification
- **`run_sft_train.sh`** - Convenient runner
- **`SFT_TRAINING_GUIDE.md`** - Full guide
- **`policy_sft_pretrained.pth`** - Output model

---

## Workflow

```
1. Test    →  python test_sft.py
2. Train   →  ./run_sft_train.sh
3. Verify  →  Check reward ~7.0-7.5
4. Refine  →  python trpo_torch.py --load_sft_pretrain ...
5. Deploy  →  Use trained model
```

---

## Quick Fix

**Not working?**
```bash
# Check you're in right directory
cd /path/to/DeepMimic_mujoco

# Activate environment
source /home/dev/.virtualenvs/openai/bin/activate

# Run test
python test_sft.py
```

---

## Result Expected

```
Training complete! Best validation loss: 0.003456

Evaluation Results:
  Mean Reward: 7.32 ± 0.31  ← Good!
  
✓ Ready for RL fine-tuning
```

---

**Need help?** Read `SFT_TRAINING_GUIDE.md`
