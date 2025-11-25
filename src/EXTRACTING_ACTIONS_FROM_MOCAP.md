# How to Extract Ground Truth Actions from Mocap Data

## Overview

Motion capture (mocap) data provides **kinematic trajectories** (positions and velocities), but **NOT actions**. We need to compute what actions would produce these movements.

---

## Method 1: PD Control (Current Implementation) ✅

### Concept
Use a **Proportional-Derivative (PD) controller** to compute actions that drive the current state toward the next mocap frame.

### Implementation (train_sft.py, lines 44-73)

```python
def compute_action_pd_control(self, qpos_current, qvel_current, qpos_target):
    """
    Compute action using PD control
    
    Action formula: action = Kp * (target - current) - Kd * velocity
    """
    kp = 1.0  # Proportional gain
    kd = 0.1  # Derivative gain
    
    # Extract joint angles (exclude root position/orientation)
    current_joints = qpos_current[7:]  # [28] - skip 3D pos + 4D quat
    target_joints = qpos_target[7:]    # [28]
    joint_vels = qvel_current[6:]      # [28] - skip 3D linear + 3D angular vel
    
    # PD control
    position_error = target_joints - current_joints
    action = kp * position_error - kd * joint_vels
    
    # Clip to action space bounds
    action = np.clip(action, -1.0, 1.0)
    
    return action
```

### How It Works

```
Frame i → Frame i+1

qpos[i]:  [x, y, z, qw, qx, qy, qz, j1, j2, ..., j28]
                                    ^^^^^^^^^^^^^^^^^
                                    Current joint angles

qpos[i+1]: [x', y', z', qw', qx', qy', qz', j1', j2', ..., j28']
                                             ^^^^^^^^^^^^^^^^^^^^
                                             Target joint angles

action = Kp * (target - current) - Kd * velocity
       = 1.0 * (j1' - j1) - 0.1 * vel1,  for joint 1
       = 1.0 * (j2' - j2) - 0.1 * vel2,  for joint 2
       ...
```

### Extraction Flow (lines 91-106)

```python
for i in range(num_frames - lookahead_frames):
    # 1. Get current state at frame i
    qpos_current = mocap.data_config[i]    # [35]
    qvel_current = mocap.data_vel[i]       # [34]
    
    # 2. Get target state at frame i+1 (or i+lookahead)
    qpos_target = mocap.data_config[i + lookahead_frames]
    
    # 3. Compute observation
    env.set_state(qpos_current, qvel_current)
    obs = env._get_obs()  # [56] - processed observation
    
    # 4. Compute action using PD control
    action = compute_action_pd_control(
        qpos_current, qvel_current, qpos_target
    )  # [28]
    
    # 5. Store pair
    dataset.append((obs, action))
```

### Pros & Cons

✅ **Pros:**
- Simple and intuitive
- Works well for smooth mocap data
- No need for inverse dynamics
- Fast to compute

❌ **Cons:**
- Actions are **approximate** (not true physical actions)
- Kp, Kd gains need tuning
- May not work for rapid movements
- Ignores dynamics (forces, torques)

---

## Method 2: Inverse Dynamics (Physically Accurate)

### Concept
Compute the **exact torques** needed to produce the mocap accelerations using inverse dynamics.

### Physics

```
Forward dynamics:  τ (torques) → q̈ (accelerations)
Inverse dynamics:  q̈ (accelerations) → τ (torques)

τ = M(q)q̈ + C(q,q̇) + G(q)

Where:
- M(q): Inertia matrix
- C(q,q̇): Coriolis/centrifugal forces  
- G(q): Gravity forces
```

### Implementation

```python
def compute_action_inverse_dynamics(self, qpos, qvel, qacc):
    """
    Compute true torques using inverse dynamics
    
    Args:
        qpos: Joint positions [35]
        qvel: Joint velocities [34]
        qacc: Joint accelerations [34] - computed from mocap
        
    Returns:
        action: Joint torques [28]
    """
    # Set MuJoCo state
    self.env.sim.data.qpos[:] = qpos
    self.env.sim.data.qvel[:] = qvel
    self.env.sim.data.qacc[:] = qacc
    
    # Compute inverse dynamics
    mujoco.mj_inverse(self.env.sim.model, self.env.sim.data)
    
    # Extract actuator forces (skip free joint)
    torques = self.env.sim.data.qfrc_inverse[6:]  # [28]
    
    # Normalize to action space [-1, 1]
    # Assuming actuators have gear ratios defined
    action = torques / self.env.sim.model.actuator_gear[:, 0]
    action = np.clip(action, -1.0, 1.0)
    
    return action
```

### Computing Accelerations from Mocap

```python
def compute_accelerations(self, mocap_data):
    """Compute accelerations from position/velocity data"""
    dt = mocap_data.dt  # Timestep
    
    accelerations = []
    for i in range(len(mocap_data.data_vel) - 1):
        qvel_current = mocap_data.data_vel[i]
        qvel_next = mocap_data.data_vel[i + 1]
        
        # Finite difference: qacc = (qvel_next - qvel_current) / dt
        qacc = (qvel_next - qvel_current) / dt
        accelerations.append(qacc)
    
    return np.array(accelerations)
```

### Pros & Cons

✅ **Pros:**
- **Physically accurate** torques
- Best for dynamic movements
- Works for all speeds

❌ **Cons:**
- More complex to implement
- Requires clean mocap data (noise in accelerations!)
- Computationally expensive
- Need to handle MuJoCo constraints

---

## Method 3: Target Positions (Joint Space Targets)

### Concept
If your actuators use **position control** (not torque control), actions ARE target positions.

### Implementation

```python
def compute_action_position_targets(self, qpos_current, qpos_target):
    """
    For position-controlled actuators
    Action = target joint position
    """
    # Extract joint angles (skip root)
    target_joints = qpos_target[7:]  # [28]
    
    # Normalize to [-1, 1] if needed
    # (depends on your action space definition)
    action = target_joints / np.pi  # Example normalization
    action = np.clip(action, -1.0, 1.0)
    
    return action
```

### Pros & Cons

✅ **Pros:**
- Simplest method
- Exact if actuators are position-controlled

❌ **Cons:**
- Only works for position-controlled actuators
- Most RL environments use torque control

---

## Method 4: Data Augmentation (Multiple Lookaheads)

### Concept
Extract multiple action sequences with different lookahead distances to create richer dataset.

### Implementation

```python
def extract_dataset_augmented(self, lookahead_frames=[1, 2, 3]):
    """Extract with multiple lookahead distances"""
    dataset = []
    
    for lookahead in lookahead_frames:
        for i in range(len(self.mocap.data_config) - lookahead):
            qpos_current = self.mocap.data_config[i]
            qvel_current = self.mocap.data_vel[i]
            qpos_target = self.mocap.data_config[i + lookahead]
            
            obs = self.get_observation(qpos_current, qvel_current)
            action = self.compute_action_pd_control(
                qpos_current, qvel_current, qpos_target
            )
            
            dataset.append((obs, action))
    
    return dataset
```

### Benefits

✅ More training data (3x with [1,2,3])
✅ Policy learns to handle different planning horizons
✅ Better generalization

---

## Your Current Pipeline (train_sft.py)

```python
# Step 1: Load mocap (lines 38-40)
self.mocap = MocapDM()
self.mocap.load_mocap(mocap_path)
# Result: 
#   self.mocap.data_config: qpos sequences [N, 35]
#   self.mocap.data_vel:    qvel sequences [N, 34]

# Step 2: Extract dataset (lines 91-111)
for i in range(num_frames - lookahead_frames):
    qpos_current = self.mocap.data_config[i]
    qvel_current = self.mocap.data_vel[i]
    qpos_target = self.mocap.data_config[i + lookahead_frames]
    
    env.set_state(qpos_current, qvel_current)
    obs = env._get_obs()  # [56] - includes derived features
    
    action = compute_action_pd_control(...)  # [28]
    
    dataset.append((obs, action))

# Step 3: Training (lines 169-193)
for batch in dataloader:
    obs_batch = [x[0] for x in batch]     # [batch_size, 56]
    action_batch = [x[1] for x in batch]  # [batch_size, 28]
    
    # Train policy to predict action from obs
    predicted_actions = policy(obs_batch)
    loss = MSE(predicted_actions, action_batch)
    loss.backward()
    optimizer.step()
```

---

## Comparison Table

| Method | Accuracy | Speed | Complexity | Best For |
|--------|----------|-------|------------|----------|
| **PD Control** | ★★★☆☆ | ★★★★★ | ★★☆☆☆ | Smooth, slow movements |
| **Inverse Dynamics** | ★★★★★ | ★★☆☆☆ | ★★★★☆ | Dynamic, fast movements |
| **Position Targets** | ★★★★★ | ★★★★★ | ★☆☆☆☆ | Position-controlled robots |
| **Augmentation** | ★★★★☆ | ★★★☆☆ | ★★★☆☆ | Limited mocap data |

---

## Tuning PD Control (Current Method)

### Adjusting Gains

```python
# In compute_action_pd_control():

# For SLOW movements (smooth tracking):
kp = 0.5   # Lower gain
kd = 0.2   # Higher damping

# For FAST movements (aggressive tracking):
kp = 2.0   # Higher gain
kd = 0.05  # Lower damping

# For DANCE (balanced):
kp = 1.0   # Current value ✓
kd = 0.1   # Current value ✓
```

### Testing Different Lookaheads

```python
# Train with different prediction horizons:

# Short-term (reactive):
python train_sft.py --lookahead 1  # Predict 1 frame ahead (0.033s)

# Medium-term (smooth):
python train_sft.py --lookahead 3  # Predict 3 frames ahead (0.1s)

# Long-term (planning):
python train_sft.py --lookahead 5  # Predict 5 frames ahead (0.166s)
```

---

## Recommendations

### For Your Dancing Task:

✅ **Stick with PD Control** (your current method)
- Dance movements are relatively smooth
- Kp=1.0, Kd=0.1 are reasonable starting points
- Fast and works well in practice

### Potential Improvements:

1. **Try different lookaheads**:
   ```bash
   python train_sft.py --lookahead 2  # Smoother predictions
   ```

2. **Add noise for robustness**:
   ```python
   action = action + np.random.normal(0, 0.01, action.shape)
   ```

3. **Multiple mocap files** for diversity:
   ```python
   dataset = []
   for mocap_file in ['dance_a.txt', 'dance_b.txt', ...]:
       dataset.extend(extract_dataset(mocap_file))
   ```

---

## Summary

**Your current implementation (PD Control) is appropriate for dancing!**

The "ground truth" actions are computed as:
```python
action = Kp * (next_pose - current_pose) - Kd * current_velocity
```

This gives the control signal that would drive the character from the current mocap frame to the next one. While not physically exact, it's a good approximation for supervised learning and works well in practice for smooth motions like dancing.
