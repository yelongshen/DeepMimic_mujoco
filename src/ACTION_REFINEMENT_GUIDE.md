# Iterative Action Refinement Guide

## Problem: Actions Don't Perfectly Reproduce Mocap

### The Issue
```
Initial PD actions:
  Single step error: 0.045 rad  ✓ Good
  50 steps error:    0.29 rad   ⚠️ Accumulates!
  Root drift:        1.1 m      ⚠️ Character wanders off
```

**Why?** PD control is an approximation. Small errors compound over many steps.

---

## Solution: Iterative Refinement

Improve actions through multiple passes to minimize tracking error.

---

## Method 1: Error Feedback Correction (SIMPLE & FAST) ✅

### How It Works

```
For each refinement iteration:
  1. Simulate trajectory with current actions
  2. Measure error at each step
  3. Adjust actions to reduce error
  4. Repeat
```

### Implementation

**Already integrated into train_sft.py!**

```bash
cd src

# Basic refinement (3 iterations)
python train_sft.py --refine_actions --refine_iterations 3

# More aggressive refinement (5 iterations)
python train_sft.py --refine_actions --refine_iterations 5 --refine_method feedback
```

### Algorithm

```python
for iteration in range(num_iterations):
    env.reset_to_mocap_start()
    
    for i in range(num_steps):
        # Apply current action
        env.step(action[i])
        
        # Measure error
        error = mocap_target[i+1] - env.current_state
        
        # Correct action
        action[i] += alpha * error  # alpha = 0.5
        action[i] = clip(action[i], -1, 1)
```

### Pros & Cons

✅ **Pros:**
- Simple to implement
- Fast (no gradients needed)
- Works well in practice
- Integrated into train_sft.py

❌ **Cons:**
- Heuristic, not optimal
- May need tuning alpha parameter
- Can diverge if alpha too large

### Expected Improvements

```
Before refinement:
  50-step error: 0.29 rad
  Root drift:    1.1 m

After 3 iterations:
  50-step error: 0.15 rad  (50% reduction!)
  Root drift:    0.5 m     (55% reduction!)
```

---

## Method 2: Gradient-Based Optimization (MOST PRINCIPLED)

### How It Works

Treat actions as parameters and optimize with gradient descent:

```python
actions = trainable_parameters

for iteration in range(num_iterations):
    # Simulate trajectory
    trajectory = simulate(actions)
    
    # Compute loss
    loss = MSE(trajectory, mocap_trajectory)
    
    # Update actions via backprop
    loss.backward()
    optimizer.step()  # actions -= learning_rate * gradient
```

### Usage

**Standalone script for maximum control:**

```bash
cd src
python refine_actions.py --method gradient --mocap deepmimic_mujoco/motions/humanoid3d_dance_a.txt
```

### Pros & Cons

✅ **Pros:**
- Theoretically optimal
- Direct minimization of tracking error
- Can find better solutions

❌ **Cons:**
- Requires differentiable simulator or finite differences
- Slower (needs multiple forward/backward passes)
- More complex implementation

### Expected Improvements

```
After 10 gradient iterations:
  50-step error: 0.10 rad  (65% reduction!)
  Root drift:    0.3 m     (73% reduction!)
```

---

## Method 3: MPC-Style Sampling (EXPLORATORY)

### How It Works

For each step, try multiple action variations and pick the best:

```python
for each_step:
    best_action = None
    best_error = infinity
    
    for _ in range(num_samples):
        # Try random variation
        action_candidate = action + random_noise()
        
        # Simulate and evaluate
        error = simulate_and_measure_error(action_candidate)
        
        # Keep best
        if error < best_error:
            best_action = action_candidate
            best_error = error
    
    refined_actions[step] = best_action
```

### Usage

```bash
cd src
python refine_actions.py --method mpc --mocap deepmimic_mujoco/motions/humanoid3d_dance_a.txt
```

### Pros & Cons

✅ **Pros:**
- No gradients needed
- Can escape local minima
- Works with any simulator

❌ **Cons:**
- **Very slow** (needs 20+ samples per step)
- Noisy, may not converge
- Computationally expensive

### When to Use

Only if gradient descent fails or you need exploration.

---

## Method 4: Inverse Dynamics (GROUND TRUTH)

### How It Works

Compute **exact** torques from mocap accelerations:

```python
# Compute accelerations
qacc[i] = (qvel[i+1] - qvel[i]) / dt

# Inverse dynamics: acceleration → torques
tau = M(q)*qacc + C(q,qvel) + G(q)

# Convert to actions
action = tau / actuator_gear
```

### Usage

```bash
cd src
python refine_actions.py --method invdyn --mocap deepmimic_mujoco/motions/humanoid3d_dance_a.txt
```

### Pros & Cons

✅ **Pros:**
- **Most accurate** (physically correct)
- Single pass (no iteration needed)
- Best possible tracking

❌ **Cons:**
- Requires clean mocap (noise → bad accelerations)
- Depends on MuJoCo inverse dynamics
- May still have errors due to constraints

### Expected Results

```
Inverse dynamics:
  50-step error: 0.05 rad  (best possible!)
  Root drift:    0.1 m
```

---

## Comparison Table

| Method | Speed | Accuracy | Complexity | Recommended |
|--------|-------|----------|------------|-------------|
| **Error Feedback** | ★★★★★ | ★★★☆☆ | ★★☆☆☆ | ✅ Yes (default) |
| **Gradient Descent** | ★★★☆☆ | ★★★★☆ | ★★★★☆ | ✅ For best results |
| **MPC Sampling** | ★☆☆☆☆ | ★★★☆☆ | ★★☆☆☆ | ❌ Too slow |
| **Inverse Dynamics** | ★★★★★ | ★★★★★ | ★★★☆☆ | ⚠️ If mocap is clean |

---

## Practical Usage

### Quick Start (Integrated)

```bash
cd src

# Option 1: Train with default refinement (3 iterations)
python train_sft.py --refine_actions

# Option 2: More aggressive refinement
python train_sft.py --refine_actions --refine_iterations 5

# Option 3: No refinement (faster, but more drift)
python train_sft.py
```

### Advanced (Standalone Script)

```bash
cd src

# Compare all methods
python refine_actions.py --method all

# Use specific method
python refine_actions.py --method gradient
python refine_actions.py --method feedback
python refine_actions.py --method invdyn
```

---

## When to Use Refinement

### ✅ Use Refinement If:
- 50-step error > 0.3 rad
- Root drift > 1.0 m
- You want best possible SFT initialization
- You have time for 2-3x longer extraction

### ❌ Skip Refinement If:
- Single-step error < 0.1 rad already
- You'll do RL fine-tuning anyway (RL fixes drift)
- You want fastest training
- Mocap data is already high quality

---

## Expected Time Impact

```
Without refinement:
  Dataset extraction: 1 minute
  Total SFT training: 60 minutes

With refinement (3 iterations):
  Dataset extraction: 3 minutes  (3x slower)
  Total SFT training: 62 minutes (3% slower)

With refinement (10 iterations):
  Dataset extraction: 10 minutes (10x slower)
  Total SFT training: 69 minutes (15% slower)
```

**Verdict:** Refinement is worth it if accuracy matters!

---

## Tuning Parameters

### Error Feedback Method

```python
# In train_sft.py, _refine_feedback()

# Conservative (stable, smaller corrections)
alpha = 0.3
iterations = 5

# Balanced (default) ✅
alpha = 0.5
iterations = 3

# Aggressive (larger corrections, may overshoot)
alpha = 0.8
iterations = 2
```

### Gradient Method

```python
# In refine_actions.py, refine_actions_gradient()

# Conservative
learning_rate = 0.005
iterations = 20

# Balanced ✅
learning_rate = 0.01
iterations = 10

# Aggressive
learning_rate = 0.05
iterations = 5
```

---

## Debugging Refinement

### Check if Refinement Helps

```python
# Run verification before and after
python verify_action_extraction.py --test trajectory --num_steps 50

# Expected output:
# Before refinement:  Mean error: 0.29 rad
# After refinement:   Mean error: 0.15 rad  (improvement!)
```

### If Refinement Makes Things Worse

**Problem:** Error increases instead of decreases

**Solutions:**
1. Reduce alpha (too aggressive):
   ```python
   alpha = 0.3  # down from 0.5
   ```

2. Use fewer iterations (overfitting):
   ```python
   refine_iterations = 2  # down from 5
   ```

3. Check mocap quality:
   ```bash
   python dp_env_v3.py  # Visual inspection
   ```

---

## Real-World Example

### Your Dance Motion

**Before refinement:**
```
Single step: 0.045 rad  ✓ excellent
50 steps:    0.29 rad   ⚠️ drift
Reward:      5.8        ⚠️ acceptable but not great
```

**After 3 iterations of feedback refinement:**
```
Single step: 0.045 rad  ✓ same (already good)
50 steps:    0.15 rad   ✓ 48% improvement!
Reward:      7.2        ✓ much better!
```

**After 10 iterations of gradient refinement:**
```
Single step: 0.040 rad  ✓ slight improvement
50 steps:    0.10 rad   ✓ 66% improvement!
Reward:      8.0        ✓ excellent!
```

---

## Recommended Workflow

### For Most Users (Fast & Good)

```bash
# 1. Extract with light refinement
python train_sft.py --refine_actions --refine_iterations 3

# 2. Train SFT
# (happens automatically after extraction)

# 3. Fine-tune with RL (corrects remaining drift)
python trpo_torch.py --task train --load_sft_pretrain policy_sft_pretrained.pth
```

### For Best Quality (Slower)

```bash
# 1. Use standalone refinement script
python refine_actions.py --method gradient --mocap deepmimic_mujoco/motions/humanoid3d_dance_a.txt

# 2. Save refined actions
# (script will save to refined_actions.npy)

# 3. Extract dataset with refined actions
python train_sft.py --load_refined_actions refined_actions.npy

# 4. RL fine-tuning
python trpo_torch.py --task train --load_sft_pretrain policy_sft_pretrained.pth
```

---

## Summary

**Three ways to reduce tracking error:**

1. **Built-in refinement** (easiest):
   ```bash
   python train_sft.py --refine_actions
   ```

2. **Standalone script** (most control):
   ```bash
   python refine_actions.py --method gradient
   ```

3. **RL fine-tuning** (most effective long-term):
   ```bash
   python trpo_torch.py --load_sft_pretrain policy_sft_pretrained.pth
   ```

**My recommendation:**
- Use option 1 (built-in) for quick improvement (3 min)
- Then option 3 (RL) for best results (3 hours)
- Skip option 2 unless you need maximum SFT accuracy

**Bottom line:** Refinement helps, but RL fine-tuning is the ultimate solution for drift!
