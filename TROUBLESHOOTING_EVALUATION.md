# Troubleshooting: Poor Evaluation Performance 🔧

## Issues Found

### 1. ⚠️ **CRITICAL: Reward Function is Disabled!**

In `dp_env_v3.py` line 243:

```python
# reward = self.calc_config_reward()  # ← Motion imitation reward (DISABLED!)
reward = reward_alive                 # ← Just returns 1.0 (ACTIVE)
```

**This means:**
- Policy is NOT being trained to imitate motion
- Policy only learns to stay alive (z_com between 0.7 and 2.0)
- No incentive to follow the reference mocap trajectory

### 2. 🐛 **Short Evaluation**

Previously had only 2 trajectories, now changed to 10 for better statistics.

### 3. 📊 **Missing Diagnostics**

Added detailed logging to verify checkpoint loading and per-trajectory results.

---

## Solutions

### Option A: Enable Motion Imitation Reward (RECOMMENDED)

Enable the proper reward that compares agent motion to reference mocap:

1. **Edit `src/dp_env_v3.py` line 243:**

```python
# Change from:
reward = reward_alive

# To:
reward = self.calc_config_reward()
```

2. **Retrain the policy** (required if trained with wrong reward):

```bash
cd /mnt/c/DeepMimic_mujoco/src
python trpo.py --task train --num_timesteps 5000000
```

### Option B: Enable Full Reward Function

For best results, use the complete reward that includes:
- Motion imitation
- Action penalty  
- Forward progress

1. **Edit `src/dp_env_v3.py` lines 236-248:**

```python
def step(self, action):
    self.step_len = 1
    step_times = 1
    pos_before = mass_center(self.model, self.sim)
    self.do_simulation(action, step_times)
    pos_after = mass_center(self.model, self.sim)

    observation = self._get_obs()

    reward_alive = 1.0
    reward_obs = self.calc_config_reward()  # Motion imitation
    reward_acs = -0.1 * np.square(self.sim.data.ctrl).sum()  # Action penalty
    reward_forward = 0.25 * (pos_after - pos_before)  # Forward movement

    reward = reward_obs + reward_acs + reward_forward + reward_alive

    info = dict(reward_obs=reward_obs, reward_acs=reward_acs, reward_forward=reward_forward)
    done = self.is_done()

    return observation, reward, done, info
```

2. **Retrain with proper reward**

---

## Quick Diagnostic

### Check if checkpoint was trained with correct reward:

```bash
cd /mnt/c/DeepMimic_mujoco/src
export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL="glfw"

# Evaluate with diagnostics
xvfb-run -a python trpo.py --task evaluate \
  --load_model_path checkpoint_tmp/DeepMimic/trpo-walk-0/DeepMimic/trpo-walk-0 \
  --stochastic_policy
```

**Look for:**
- "Policy test - Action shape: ..." ← Verifies model loads
- Per-trajectory returns ← Should vary if using motion reward
- Average return ← Should be > 1.0 if using motion reward

**If all returns are close to 1.0:**
→ Policy was trained with `reward_alive` only
→ Need to retrain with proper reward

---

## What Changed in trpo.py

### Evaluation Parameters:
```python
# Before:
timesteps_per_batch=1024,
number_trajs=2,

# After:
timesteps_per_batch=2048,  # Longer episodes
number_trajs=10,           # More episodes for statistics
```

### Added Diagnostics:
```python
✓ Checkpoint loading verification
✓ Test action to verify policy works
✓ Per-trajectory statistics (first 3 + last)
✓ Statistics: mean ± std, min, max
✓ Better formatted output
✓ Motion name in video filename
```

---

## Expected Behavior

### With Motion Imitation Reward:
```
📊 RESULTS:
  Average episode length: 150.23 ± 45.2
  Average return:         45.67 ± 12.3  ← Higher than 1.0
  Min return:             28.45
  Max return:             65.23
```

### With Only Alive Reward (Current):
```
📊 RESULTS:
  Average episode length: 120.00 ± 5.0
  Average return:         1.20 ± 0.1    ← Close to 1.0
  Min return:             1.10
  Max return:             1.30
```

---

## Recommended Action Plan

1. **✅ Run diagnostic evaluation** (see above) to confirm issue

2. **🔧 Enable proper reward** in `dp_env_v3.py`:
   ```python
   reward = self.calc_config_reward()
   ```

3. **🏋️ Retrain policy** from scratch:
   ```bash
   cd src
   python trpo.py --task train --num_timesteps 5000000 --seed 0
   ```

4. **🎬 Re-evaluate** with new checkpoint:
   ```bash
   ./run_eval_hide_cuda.sh
   ```

5. **👀 Watch video** in `./render/` to verify motion quality

---

## Training Tips

### For better motion imitation:

1. **Increase motion weight**:
   ```python
   self.weight_pose = 0.65  # Up from 0.5
   ```

2. **Use appropriate timesteps**:
   ```bash
   --num_timesteps 10000000  # 10M for complex motions
   ```

3. **Monitor training**:
   ```bash
   tensorboard --logdir log_tmp/
   ```

4. **Check reward components**:
   - Add logging to see reward_obs, reward_acs, etc.
   - Motion reward should dominate

---

## Summary

**Current Issue:** Reward is hardcoded to 1.0, so policy doesn't learn motion imitation.

**Quick Fix:** Change line 243 in `dp_env_v3.py` to use `self.calc_config_reward()`

**Full Fix:** Retrain policy with proper reward function enabled.

**Verification:** Returns should be >> 1.0 and video should show motion matching reference.

Good luck! 🚀
