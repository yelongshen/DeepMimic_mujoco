# TRPO Policy Not Updating - Debug Guide

## Symptom

```
Policy loss: -0.0000
KL divergence: 0.000000
```

Both are exactly zero, meaning the policy is not updating at all.

## Possible Causes & Diagnostics

### 1. Advantages are Zero/Constant

**Check:** Look at debug output for:
```
Advantages (raw): mean=X, std=X
```

**Problem:** If `std ≈ 0`, all advantages are the same → no gradient signal

**Why:** 
- Rewards are constant (all same value)
- Value function perfectly predicts returns (V(s) = actual return always)
- GAE λ is 1.0 (no temporal difference)

**Fix:**
- Verify rewards vary (check min/max rewards)
- Check if value function is overfitting
- Reduce GAE lambda from 0.97 to 0.95

---

### 2. Gradient Norm is Zero

**Check:** Look for:
```
Gradient norm: 0.000000
```

**Problem:** No policy gradient despite non-zero advantages

**Why:**
- Log probabilities are identical before/after action
- Policy is deterministic (std=0)
- Numerical precision issues

**Fix:**
- Check `ratio_mean` and `ratio_std` - should not be 1.0 exactly
- Verify policy std is > 0 (check entropy - should be moderate, not huge)
- Check if policy parameters have gradients enabled

---

### 3. Step Direction Norm is Zero

**Check:** Look for:
```
Step direction norm: 0.000000
```

**Problem:** Conjugate gradient failed to find direction

**Why:**
- Fisher matrix is singular
- Damping too low
- CG iterations insufficient

**Fix:**
- Increase damping from 0.1 to 0.5
- Increase CG iterations from 10 to 20
- Check that Fvp computation is correct

---

### 4. Full Step Norm is Zero

**Check:** Look for:
```
Full step norm: 0.000000
```

**Problem:** Step size calculation went to zero

**Why:**
- max_kl too small
- Expected improvement is zero/negative
- Numerical overflow in step size computation

**Fix:**
- Increase max_kl from 0.01 to 0.05
- Check expected_improve value

---

### 5. Line Search Always Fails

**Check:** Look for:
```
Line search: ✗ Failed
```

**Problem:** No step size satisfies KL constraint and improvement

**Why:**
- KL divergence too large even for smallest step
- Expected improvement is negative
- Reward computation is broken

**Fix:**
- Reduce step size (increase damping)
- Check that rewards are correct
- Verify KL computation

---

### 6. Policy is Already Optimal (Unlikely)

**Check:** Is reward near maximum (10.0)?

**Problem:** Agent already solved the task

**Why:** Very unlikely with random initialization

**Fix:** N/A - task is solved!

---

## Debug Output Interpretation

### Healthy Training (Example)

```
Iteration 10 | Timesteps 20480
  Avg episode reward: 2.5432
  Min/Max reward: 0.34 / 5.67
  Advantages - mean: 0.000000, std: 0.234567  ← Good variance!
  Returns - mean: 2.543, std: 1.234
  Policy loss: -0.0123
  KL divergence: 0.008123  ← Policy changing!
  Entropy: 12.4567  ← Moderate (stochastic policy)
  Value loss: 0.4567
  Line search: ✓ Success

  [DEBUG]
    Advantages (raw): mean=0.012345, std=0.234567  ← Non-zero!
    Advantages range: [-0.8765, 1.2345]
    Ratio: mean=1.000123, std=0.012345  ← Close to 1.0 but varying
    Gradient norm: 0.045678  ← Non-zero gradient!
    Step direction norm: 3.456789
    Full step norm: 0.012345  ← Small but non-zero
    Expected improvement: 0.000567  ← Positive!
```

### Broken Training (What You're Seeing)

```
Iteration 10 | Timesteps 20480
  Avg episode reward: 0.0001  ← Too small!
  Min/Max reward: 0.0001 / 0.0001  ← All the same!
  Advantages - mean: 0.000000, std: 0.000000  ← No variance! PROBLEM!
  Returns - mean: 0.0001, std: 0.000000
  Policy loss: -0.0000  ← Zero
  KL divergence: 0.000000  ← Zero
  Entropy: 39.5  ← Too high (nearly random)
  Value loss: 0.0000  ← Zero
  Line search: ✗ Failed

  [DEBUG]
    Advantages (raw): mean=0.000000, std=0.000000  ← PROBLEM HERE!
    Advantages range: [0.0000, 0.0000]
    Ratio: mean=1.000000, std=0.000000  ← No change
    Gradient norm: 0.000000  ← Zero gradient!
    Step direction norm: 0.000000  ← Can't find direction
    Full step norm: 0.000000  ← No step
    Expected improvement: 0.000000  ← No improvement expected
```

---

## Quick Fixes to Try

### Fix 1: Verify Reward Scaling is Applied

```bash
cd /mnt/c/DeepMimic_mujoco/src
python -c "
import sys; sys.path.append('env')
from dp_env_v3 import DPEnv
env = DPEnv()
ob = env.reset(); ob = env.reset_model_init()
for i in range(20):
    ob, r, d, _ = env.step(env.action_space.sample())
    print(f'Step {i+1}: reward={r:.4f}')
"
```

**Expected:** Rewards between 0.01 and 5.0  
**Problem:** All rewards exactly the same or near zero

### Fix 2: Increase Reward Variance

Edit `dp_env_v3.py`:

```python
def calc_config_reward(self):
    # ... existing code ...
    
    # Add noise to break ties if rewards are too similar
    noise = np.random.normal(0, 0.01)  # Small noise
    reward_config = reward_config + noise
    
    return max(0.0, reward_config)  # Clip to positive
```

### Fix 3: Reduce Value Function Overfitting

In `trpo_torch.py`, reduce value function iterations:

```python
# Change from:
vf_iters=3

# To:
vf_iters=1
```

This prevents value function from perfectly fitting and leaving no advantages.

### Fix 4: Increase TRPO Hyperparameters

```python
TRPOAgent(
    env, policy,
    max_kl=0.05,      # Increase from 0.01
    damping=0.5,      # Increase from 0.1
    gamma=0.995,
    lam=0.95,         # Decrease from 0.97 for more variance
    vf_iters=1,       # Decrease from 3
    vf_lr=1e-3,
    cg_iters=20,      # Increase from 10
    device='cpu'
)
```

---

## Action Plan

1. **Run training with debug output** (already enabled)
2. **Check which diagnostic value is zero:**
   - If `Advantages std = 0` → rewards are constant (check reward function)
   - If `Gradient norm = 0` → policy has no gradient (check policy architecture)
   - If `Step direction norm = 0` → increase damping/CG iters
   - If `Full step norm = 0` → increase max_kl

3. **Apply corresponding fix** from above

4. **Verify fix** by checking debug output shows non-zero values

---

## Expected Timeline

- **After reward scaling fix:** Rewards should be 0.1 - 5.0
- **After hyperparameter fix:** Debug values should be non-zero
- **After 10 iterations:** Should see policy starting to update (KL > 0)
- **After 100 iterations:** Reward should improve noticeably

If still not working after these fixes, there may be a fundamental bug in the environment or policy architecture.
