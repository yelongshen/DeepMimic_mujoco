# Training Not Improving - Diagnosis & Solutions

## Observed Problem

After 360 iterations (~737K timesteps):
- **Reward stuck at 0.0001** (not improving at all)
- Policy loss: ~0.0000 (no gradient signal)
- KL divergence: 0.000000 (policy not changing)
- Value loss: 0.0000 (value function not learning)
- Entropy: ~39.5 (very high - almost random policy)

## Root Cause Analysis

### 1. Reward Scale Issue

The reward function is: `reward = exp(-error)` where error is the sum of absolute joint differences.

For a humanoid with ~29 joints (excluding root):
- Random policy: errors typically 5-20 per joint
- Total error: 5-20 × 29 = **145-580**
- Reward: exp(-145) to exp(-580) ≈ **10^-63 to 10^-252** (essentially 0!)

Your observed reward of 0.0001 = exp(-9.21) suggests total error ~9.21, which is **very good** for imitation, but the agent started there by chance and isn't improving.

### 2. No Learning Signal

With such tiny rewards:
- Advantages ≈ 0 → no policy gradient
- Returns ≈ 0 → value function has nothing to learn
- Policy stuck in initial random state

## Solutions

### Option 1: Reward Scaling (Recommended)

Multiply the reward to make it meaningful for learning:

```python
# In dp_env_v3.py, line ~226:
def calc_config_reward(self):
    # ... existing code ...
    err_configs = self.calc_config_errs(curr_config, target_config)
    
    # Scale error before exp to get reasonable rewards
    # Divide by number of joints to normalize
    num_joints = len(target_config)
    normalized_err = err_configs / num_joints
    
    # Apply exp with scaled error
    reward_config = math.exp(-5.0 * normalized_err)  # Scale factor of 5
    
    # Or use simpler linear reward for debugging:
    # reward_config = max(0, 1.0 - normalized_err)
    
    # Optionally multiply by large constant to make gradient signals stronger
    reward_config = reward_config * 10.0  # Scale up for better learning
    
    return reward_config
```

This gives rewards in range 0-10 instead of 0-0.0001.

### Option 2: Add Dense Rewards

Instead of just final pose matching, reward intermediate progress:

```python
def step(self, action):
    # ... existing code ...
    
    # Motion imitation reward (pose matching)
    reward_pose = self.calc_config_reward() * 10.0
    
    # Velocity matching reward
    target_vel = self.mocap.data_velocity[self.idx_curr][6:]
    curr_vel = self.sim.data.qvel[6:]
    err_vel = np.sum(np.abs(curr_vel - target_vel)) / len(curr_vel)
    reward_vel = np.exp(-2.0 * err_vel)
    
    # Alive bonus (encourages not falling)
    reward_alive = 1.0 if not done else 0.0
    
    # Action regularization (smooth actions)
    reward_action = -0.01 * np.sum(np.square(action))
    
    # Total reward
    reward = reward_pose + reward_vel + reward_alive + reward_action
    
    return observation, reward, done, info
```

### Option 3: Use Different Reward Function

Replace exponential with something more interpretable:

```python
def calc_config_reward(self):
    # ... existing code ...
    err_configs = self.calc_config_errs(curr_config, target_config)
    
    # Normalize by number of joints and typical error range
    num_joints = len(target_config)
    normalized_err = err_configs / (num_joints * 0.5)  # Assume 0.5 rad typical error
    
    # Linear reward (easier to interpret)
    reward_config = max(0, 1.0 - normalized_err)
    
    # Or quadratic (smoother)
    reward_config = max(0, 1.0 - normalized_err ** 2)
    
    return reward_config * 10.0  # Scale to 0-10 range
```

### Option 4: Check Initial State Distribution

The agent might be initialized too far from the reference motion:

```python
# In dp_env_v3.py, reset_model_init()
def reset_model_init(self):
    # Start closer to reference pose instead of random
    target_qpos = self.mocap.data[self.idx_init, :]
    
    # Add small noise for exploration
    noise = np.random.normal(0, 0.1, size=target_qpos.shape)
    init_qpos = target_qpos + noise
    
    self.set_state(init_qpos, self.init_qvel)
    return self._get_obs()
```

## Diagnostic Steps

1. **Run diagnostic script:**
   ```bash
   cd /mnt/c/DeepMimic_mujoco/src
   python diagnose_training.py
   ```
   
   This will show you:
   - Typical reward values with random policy
   - What perfect imitation would give
   - Verify reward function is being called correctly

2. **Check training with enhanced logging:**
   
   The updated `trpo_torch.py` now prints:
   - Min/Max rewards per batch
   - Advantage statistics (mean/std)
   - Return statistics
   - Line search success/failure
   
   Run training again and look for:
   - Are advantages non-zero?
   - Are returns meaningful?
   - Is line search succeeding?

3. **Verify environment:**
   ```bash
   cd /mnt/c/DeepMimic_mujoco/src
   python -c "
   import sys; sys.path.append('env')
   from dp_env_v3 import DPEnv
   env = DPEnv()
   ob = env.reset()
   ob = env.reset_model_init()
   for i in range(10):
       ob, r, d, _ = env.step(env.action_space.sample())
       print(f'Step {i+1}: reward={r:.6f}')
   "
   ```

## Recommended Fix (Quick Start)

Edit `src/dp_env_v3.py`, line ~226:

```python
def calc_config_reward(self):
    assert len(self.mocap.data) != 0
    err_configs = 0.0

    target_config = self.mocap.data_config[self.idx_curr][7:]
    self.curr_frame = target_config
    curr_config = self.get_joint_configs()

    err_configs = self.calc_config_errs(curr_config, target_config)
    
    # FIXED: Normalize and scale reward for better learning
    num_joints = len(target_config)
    normalized_err = err_configs / num_joints  # Per-joint average error
    reward_config = math.exp(-2.0 * normalized_err)  # Scaled exponential
    reward_config = reward_config * 10.0  # Scale to 0-10 range

    self.idx_curr += 1
    self.idx_curr = self.idx_curr % self.mocap_data_len

    return reward_config
```

Then restart training and watch the reward values with the enhanced logging.

## Expected Results After Fix

With proper reward scaling, you should see:
- Rewards in range 0.1 - 10.0 (not 0.0001)
- Advantages with std > 0.01 (not ~0)
- Policy loss changing between iterations
- KL divergence > 0 (policy is updating)
- Gradual improvement in average reward over iterations

## Summary

**Problem:** Reward too small → no learning signal → agent stuck  
**Solution:** Scale reward to reasonable range (0-10)  
**Action:** Edit `calc_config_reward()` to normalize and scale  
**Verify:** Run `diagnose_training.py` to check reward values
