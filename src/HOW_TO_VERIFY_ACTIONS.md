# How to Verify Extracted Actions Match Mocap Data

## Quick Summary

You want to verify that the PD-control-computed actions **actually reproduce** the mocap motion when applied in the environment.

---

## Method 1: Built-in Verification (EASIEST) ✅

### Now Automatic!

The `train_sft.py` now automatically verifies actions after extraction:

```bash
cd src
python train_sft.py
```

**What it checks:**
1. ✅ Action statistics (mean, std, min, max)
2. ✅ Clipping percentage (should be <20%)
3. ✅ Single-step reproduction test (frame 0→1)
4. ✅ Joint tracking error (should be <0.1 rad)
5. ✅ Root position error (should be <0.1 m)

**Example output:**
```
============================================================
Action Extraction Verification
============================================================

Action statistics (96 samples):
  Mean:  0.0234
  Std:   0.2841
  Min:   -1.0000
  Max:   1.0000
  Clipped: 234/2688 (8.7%)
  ✓ Good: Most actions within bounds

------------------------------------------------------------
Testing if actions reproduce mocap motion (frame 0→1):
------------------------------------------------------------
  Joint tracking error:  0.004523 rad (mean)
  Root position error:   0.012341 m
  Reward after 1 step:   7.2341
  ✓ PASS: Actions reproduce mocap motion well!
============================================================
```

---

## Method 2: Comprehensive Verification Script

For detailed analysis, use the verification script:

```bash
cd src
python verify_action_extraction.py --test all --num_steps 50
```

### Available Tests

#### Test 1: Single Step Test
```bash
python verify_action_extraction.py --test single --frame_idx 10
```
**Checks:** Does action at frame 10 move character toward frame 11?

#### Test 2: Trajectory Following
```bash
python verify_action_extraction.py --test trajectory --num_steps 100
```
**Checks:** Do repeated actions follow the entire mocap trajectory?

**Generates:** `action_verification_results.png` with 4 plots:
- Joint tracking error over time
- Root position error over time
- Rewards over time
- Action magnitudes over time

#### Test 3: Action Consistency
```bash
python verify_action_extraction.py --test consistency
```
**Checks:** Are actions consistent and reasonable across all frames?

---

## What to Look For

### ✅ Good Results (Actions Match Mocap)

```
Joint tracking error:  0.004-0.05 rad (mean)
Root position error:   0.01-0.05 m
Reward:                >7.0
Clipped actions:       <20%
```

**Interpretation:** Actions accurately reproduce mocap motion!

### ⚠️ OK Results (Acceptable)

```
Joint tracking error:  0.05-0.15 rad (mean)
Root position error:   0.05-0.15 m
Reward:                5.0-7.0
Clipped actions:       20-40%
```

**Interpretation:** Actions approximately track mocap, may improve with tuning.

### ❌ Poor Results (Actions Don't Match)

```
Joint tracking error:  >0.2 rad (mean)
Root position error:   >0.2 m
Reward:                <5.0
Clipped actions:       >50%
```

**Interpretation:** Actions don't reproduce mocap well. Need to adjust parameters!

---

## Troubleshooting Poor Results

### Problem 1: High Clipping (>50% actions clipped)

**Symptoms:**
```
Clipped: 1344/2688 (50.0%)
⚠️  WARNING: >50% actions clipped!
```

**Cause:** Actions too large → saturating at ±1.0

**Solutions:**
1. **Reduce Kp gain** (makes actions gentler):
   ```python
   # In train_sft.py, line 59
   kp = 0.5  # Reduce from 1.0
   ```

2. **Increase lookahead** (more time to reach target):
   ```bash
   python train_sft.py --lookahead 3  # Instead of 1
   ```

3. **Increase Kd damping** (smooth out motions):
   ```python
   # In train_sft.py, line 60
   kd = 0.2  # Increase from 0.1
   ```

### Problem 2: Large Tracking Error (>0.2 rad)

**Symptoms:**
```
Joint tracking error:  0.3456 rad (mean)
⚠️  WARNING: Poor tracking!
```

**Cause:** PD gains not tuned for this motion

**Solutions:**
1. **Increase Kp** (stronger correction):
   ```python
   kp = 1.5  # Increase from 1.0
   ```

2. **Check mocap quality** (noise in data?):
   ```python
   # Visualize mocap
   python dp_env_v3.py  # Should look smooth
   ```

3. **Try different lookahead**:
   ```bash
   # Test different values
   python train_sft.py --lookahead 1
   python train_sft.py --lookahead 2
   python train_sft.py --lookahead 3
   ```

### Problem 3: Drift Over Time

**Symptoms:**
```
Initial error: 0.02 rad
After 50 steps: 0.30 rad  # Error accumulates!
```

**Cause:** Actions don't perfectly reproduce motion → errors accumulate

**This is actually NORMAL!** Because:
- Actions are approximations (PD control, not true inverse dynamics)
- Small errors accumulate over many steps
- This is why we use RL fine-tuning after SFT!

**Solutions:**
1. **Use shorter trajectories for SFT**
2. **Fine-tune with RL** (corrects accumulated drift):
   ```bash
   python trpo_torch.py --load_sft_pretrain policy_sft_pretrained.pth
   ```

---

## Interpreting the Plots

### Plot 1: Joint Tracking Error
```
Good:    Stays below 0.1 rad
OK:      Increases slowly to 0.2 rad
Bad:     Rapidly increases to >0.5 rad
```

### Plot 2: Root Position Error
```
Good:    Stays below 0.1 m
OK:      Increases slowly to 0.2 m
Bad:     Character drifts far from mocap path
```

### Plot 3: Rewards
```
Good:    Consistently >7.0
OK:      Fluctuates 5.0-7.0
Bad:     Drops below 5.0
```

### Plot 4: Action Magnitudes
```
Good:    Mean ~0.2-0.5, max ~0.8
OK:      Mean ~0.5-0.8, max ~1.0
Bad:     Constantly at 1.0 (saturated)
```

---

## Manual Verification (Advanced)

### Verify Single Action Manually

```python
from dp_env_v3 import DPEnv
from deepmimic_mujoco.mocap_v2 import MocapDM
import numpy as np

# Load mocap
mocap = MocapDM()
mocap.load_mocap('deepmimic_mujoco/motions/humanoid3d_dance_a.txt')

# Get states
qpos_0 = mocap.data_config[0]
qvel_0 = mocap.data_vel[0]
qpos_1 = mocap.data_config[1]

# Compute action (PD control)
kp, kd = 1.0, 0.1
current_joints = qpos_0[7:]
target_joints = qpos_1[7:]
joint_vels = qvel_0[6:]
action = kp * (target_joints - current_joints) - kd * joint_vels
action = np.clip(action, -1.0, 1.0)

# Test action
env = DPEnv()
env.set_state(qpos_0, qvel_0)
obs, reward, done, info = env.step(action)

# Check result
qpos_actual = env.sim.data.qpos
error = np.abs(qpos_actual[7:] - qpos_1[7:])
print(f"Joint error: {error.mean():.6f} rad")
print(f"Reward: {reward:.4f}")
```

---

## PD Gain Tuning Guide

### Conservative (Smooth, less clipping)
```python
kp = 0.5
kd = 0.2
```
- Pro: Stable, <10% clipping
- Con: Slower tracking, larger errors

### Balanced (Default) ✅
```python
kp = 1.0
kd = 0.1
```
- Pro: Good trade-off
- Con: 10-20% clipping for fast motions

### Aggressive (Fast tracking, more clipping)
```python
kp = 2.0
kd = 0.05
```
- Pro: Better tracking, lower errors
- Con: 30-50% clipping, less stable

---

## When to Worry

### 🚨 Red Flags:
- ❌ Joint error >0.5 rad
- ❌ Root error >0.5 m
- ❌ Reward <3.0
- ❌ >70% actions clipped
- ❌ Action mean >0.8 or <-0.8

### ✅ Good Signs:
- ✓ Joint error <0.1 rad
- ✓ Root error <0.1 m
- ✓ Reward >7.0
- ✓ <20% actions clipped
- ✓ Actions distributed around 0

---

## Quick Checklist

Before training SFT, verify:

1. ✅ Run `python train_sft.py` → check automatic verification
2. ✅ Run `python verify_action_extraction.py --test trajectory`
3. ✅ Check plot: joint error should stay low
4. ✅ Verify: <20% actions clipped
5. ✅ Verify: reward >6.0

If all pass → proceed with SFT training!
If any fail → tune Kp/Kd or adjust lookahead.

---

## Summary

**Your extracted actions are good if:**
```
✓ Joint tracking error < 0.1 rad
✓ Root position error < 0.1 m  
✓ Reward > 6.0
✓ <20% actions clipped
✓ Plots show stable tracking
```

**Run verification:**
```bash
cd src
python train_sft.py  # Auto-verification built-in
# OR
python verify_action_extraction.py --test all
```

**Tuning knobs:**
- `kp`: Proportional gain (how hard to push toward target)
- `kd`: Derivative gain (how much to dampen velocity)
- `lookahead_frames`: How far ahead to predict

Start with defaults (kp=1.0, kd=0.1, lookahead=1) and adjust based on verification results!
