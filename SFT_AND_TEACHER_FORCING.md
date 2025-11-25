# Using SFT and Teacher Forcing for Motion Imitation

## TL;DR: Yes, and it's often better than pure RL!

**Supervised Learning from Mocap:**
- Use mocap data as ground truth labels
- Train policy to directly predict actions that reproduce mocap poses
- Much faster than RL (hours vs days)
- Often achieves better motion quality

---

## Current Approach: Reinforcement Learning (RL)

### What You're Doing Now

```python
# RL approach (TRPO):
for iteration in range(num_iterations):
    # 1. Roll out policy in environment
    obs, actions, rewards = collect_trajectories(policy, env)
    
    # 2. Compute reward based on mocap similarity
    reward = similarity(current_pose, mocap_reference_pose)
    
    # 3. Update policy to maximize reward
    policy = update_via_trpo(policy, obs, actions, rewards)
```

**Characteristics:**
- ✅ Learns through trial and error
- ✅ Can discover creative solutions
- ❌ Slow convergence (needs many episodes)
- ❌ Reward engineering is tricky
- ❌ Can get stuck in local optima

---

## Alternative: Supervised Fine-Tuning (SFT)

### Approach 1: Direct Action Prediction (Supervised Learning)

The key insight: **Mocap data tells us the optimal actions!**

```python
# Extract ground truth actions from mocap
def extract_actions_from_mocap(mocap_data):
    """
    Given mocap poses, compute what actions would produce them
    """
    actions = []
    for t in range(len(mocap_data) - 1):
        current_pose = mocap_data[t]
        next_pose = mocap_data[t + 1]
        
        # What action would move from current to next?
        action = inverse_dynamics(current_pose, next_pose)
        actions.append(action)
    
    return actions

# Supervised training
def train_sft(policy, mocap_data):
    """
    Directly train policy to predict mocap actions
    """
    observations = []
    target_actions = []
    
    for frame_t in mocap_data:
        # Get observation (joint angles, velocities)
        obs = extract_observation(frame_t)
        
        # Get target action (what should be done)
        action = extract_action(frame_t, frame_t+1)
        
        observations.append(obs)
        target_actions.append(action)
    
    # Standard supervised learning
    for epoch in range(num_epochs):
        for batch in batches(observations, target_actions):
            obs_batch, action_batch = batch
            
            # Predict actions
            predicted_actions = policy(obs_batch)
            
            # Compute loss (mean squared error)
            loss = mse_loss(predicted_actions, action_batch)
            
            # Backprop
            loss.backward()
            optimizer.step()
```

**Advantages:**
- ✅ **Much faster** - no environment rollouts needed
- ✅ **Stable** - supervised learning is well-understood
- ✅ **Direct supervision** - know exactly what to do
- ✅ **No reward engineering** - use mocap as labels

**Challenges:**
- ⚠️ Need to compute inverse dynamics (action from pose change)
- ⚠️ Distribution shift (training offline, testing online)
- ⚠️ No exploration (only learns mocap, not recovery)

---

## Approach 2: Teacher Forcing with Physics

### What is Teacher Forcing?

**Idea:** During training, use ground truth states instead of predicted states

```python
# WITHOUT teacher forcing (standard RL):
state_0 = env.reset()
action_0 = policy(state_0)
state_1 = env.step(action_0)  # Use resulting state
action_1 = policy(state_1)    # Predict from actual result
state_2 = env.step(action_1)  # Errors accumulate!

# WITH teacher forcing (supervised):
state_0 = mocap[t=0]
action_0 = policy(state_0)
state_1 = mocap[t=1]          # Use ground truth, not result!
action_1 = policy(state_1)    # Train on perfect state
state_2 = mocap[t=2]          # Always on track
```

**Benefits:**
- ✅ Prevents error accumulation during training
- ✅ Learns from high-quality state distribution
- ✅ Stable gradients

**Drawback:**
- ⚠️ Train/test mismatch (no ground truth at test time)

---

## Approach 3: Hybrid SFT + RL (Best of Both Worlds!)

### The Recommended Approach

```python
# Phase 1: Supervised Pre-training (SFT)
print("Phase 1: Learning from mocap (supervised)...")
for epoch in range(sft_epochs):
    for batch in mocap_batches:
        obs, target_actions = batch
        predicted = policy(obs)
        loss = mse_loss(predicted, target_actions)
        loss.backward()
        optimizer.step()

print("✓ Policy can now roughly imitate mocap")

# Phase 2: RL Fine-tuning (handles physics reality)
print("Phase 2: Refining with physics (RL)...")
for iteration in range(rl_iterations):
    # Start from good policy (pre-trained)
    obs, actions, rewards = collect_trajectories(policy, env)
    
    # Fine-tune with TRPO
    policy = update_via_trpo(policy, obs, actions, rewards)

print("✓ Policy robust to perturbations and realistic physics")
```

**Why this works:**
1. **SFT gets you 80% there quickly** (hours, not days)
2. **RL fine-tuning handles the last 20%** (physics realism, recovery)
3. **Best quality + efficiency**

---

## Implementation Options

### Option A: Action Regression (Simple)

```python
# Simple approach: Predict actions directly from observations
class MotionImitationSFT:
    def __init__(self, policy, mocap_data):
        self.policy = policy
        self.mocap = mocap_data
        
    def prepare_data(self):
        """Extract (observation, action) pairs from mocap"""
        dataset = []
        
        for i in range(len(self.mocap) - 1):
            # Current state
            qpos_t = self.mocap.data_config[i]
            qvel_t = self.mocap.data_vel[i]
            obs_t = self.get_obs(qpos_t, qvel_t)
            
            # Next state
            qpos_t1 = self.mocap.data_config[i + 1]
            qvel_t1 = self.mocap.data_vel[i + 1]
            
            # Compute action via PD control target
            action_t = self.compute_action(qpos_t, qvel_t, qpos_t1)
            
            dataset.append((obs_t, action_t))
        
        return dataset
    
    def compute_action(self, qpos_t, qvel_t, qpos_target):
        """
        Compute action that would move from current to target pose
        Using PD control formulation
        """
        # PD control: action = Kp * (target - current) + Kd * (-velocity)
        kp = 1.0
        kd = 0.1
        
        current_joint_pos = qpos_t[7:]  # Exclude root
        target_joint_pos = qpos_target[7:]
        joint_vel = qvel_t[6:]
        
        action = kp * (target_joint_pos - current_joint_pos) - kd * joint_vel
        
        # Clip to action bounds
        action = np.clip(action, -1, 1)
        return action
    
    def train(self, num_epochs=100, batch_size=256):
        """Train policy with supervised learning"""
        dataset = self.prepare_data()
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        
        for epoch in range(num_epochs):
            # Shuffle data
            np.random.shuffle(dataset)
            
            epoch_loss = 0
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i+batch_size]
                obs_batch = torch.tensor([x[0] for x in batch])
                action_batch = torch.tensor([x[1] for x in batch])
                
                # Forward pass
                predicted_actions = self.policy.act(obs_batch, stochastic=False)
                
                # Loss
                loss = torch.mean((predicted_actions - action_batch) ** 2)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            print(f"Epoch {epoch}: Loss = {epoch_loss / len(dataset):.6f}")
```

---

### Option B: Pose Tracking (Better)

```python
# Better approach: Learn to track target poses
class PoseTrackingSFT:
    def prepare_data(self):
        """Use current pose + target pose as input"""
        dataset = []
        
        for i in range(len(self.mocap) - 1):
            # Current observation
            obs_t = self.get_obs(self.mocap[i])
            
            # Target pose (what we want to achieve)
            target_pose = self.mocap[i + 1][7:]  # Joint angles only
            
            # Ground truth action (computed via inverse dynamics or PD)
            action_t = self.compute_tracking_action(obs_t, target_pose)
            
            # Input: obs + target
            input_t = np.concatenate([obs_t, target_pose])
            
            dataset.append((input_t, action_t))
        
        return dataset
```

---

### Option C: State Regularization (Most Robust)

```python
# Most robust: SFT with state regularization
def train_with_state_regularization(policy, mocap_data, env):
    """
    Hybrid approach: supervised loss + physics consistency
    """
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    
    for epoch in range(num_epochs):
        for batch in mocap_batches:
            obs, target_actions = batch
            
            # 1. Supervised loss (match mocap actions)
            predicted = policy(obs)
            supervised_loss = mse_loss(predicted, target_actions)
            
            # 2. Physics consistency loss (actions should make sense)
            # Roll out in environment
            next_states = env.step_batch(obs, predicted)
            expected_states = mocap_next_states[batch]
            physics_loss = mse_loss(next_states, expected_states)
            
            # 3. Combined loss
            total_loss = supervised_loss + 0.1 * physics_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
```

---

## Comparison: RL vs SFT vs Hybrid

| Aspect | Pure RL (Current) | Pure SFT | Hybrid (Recommended) |
|--------|-------------------|----------|----------------------|
| **Training time** | Days | Hours | Hours + some tuning |
| **Motion quality** | Good | Good | Best |
| **Stability** | Unstable | Very stable | Stable |
| **Robustness** | High | Low | High |
| **Implementation** | Complex | Simple | Moderate |
| **Sample efficiency** | Low | N/A (offline) | High |
| **Handles physics** | Yes | No | Yes |
| **Recovery behavior** | Yes | No | Yes |

---

## Practical Implementation for Your Codebase

### Step 1: Create SFT Training Script

```python
# src/train_sft.py

import torch
import numpy as np
from mlp_policy_torch import MlpPolicy
from dp_env_v3 import DPEnv
from deepmimic_mujoco.mocap_v2 import MocapDM

def extract_sft_dataset(mocap_path, env):
    """Extract (observation, action) pairs from mocap"""
    mocap = MocapDM()
    mocap.load_mocap(mocap_path)
    
    dataset = []
    
    for i in range(len(mocap.data) - 1):
        # Current state
        qpos = mocap.data_config[i]
        qvel = mocap.data_vel[i]
        
        # Set environment to mocap state
        env.set_state(qpos, qvel)
        obs = env._get_obs()
        
        # Target next state
        qpos_next = mocap.data_config[i + 1]
        
        # Compute action (PD control)
        kp, kd = 1.0, 0.1
        current_joints = qpos[7:]
        target_joints = qpos_next[7:]
        joint_vels = qvel[6:]
        
        action = kp * (target_joints - current_joints) - kd * joint_vels
        action = np.clip(action, -1, 1)
        
        dataset.append((obs, action))
    
    return dataset

def train_sft(policy, dataset, num_epochs=100, batch_size=256, lr=1e-3):
    """Train policy with supervised learning"""
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        np.random.shuffle(dataset)
        epoch_loss = 0
        
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            obs_batch = torch.tensor([x[0] for x in batch], dtype=torch.float32)
            action_batch = torch.tensor([x[1] for x in batch], dtype=torch.float32)
            
            # Forward
            policy_dist = policy.pd.forward(policy.pol_net(obs_batch))
            predicted_actions = policy_dist.mode()  # Mean action
            
            # Loss
            loss = torch.mean((predicted_actions - action_batch) ** 2)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {epoch_loss / (len(dataset) / batch_size):.6f}")
    
    return policy

if __name__ == "__main__":
    # Setup
    env = DPEnv()
    policy = MlpPolicy(env.observation_space, env.action_space)
    
    # Extract dataset from mocap
    print("Extracting dataset from mocap...")
    dataset = extract_sft_dataset(env.mocap_path, env)
    print(f"Dataset size: {len(dataset)} samples")
    
    # Train with SFT
    print("Training with supervised learning...")
    policy = train_sft(policy, dataset, num_epochs=100)
    
    # Save pre-trained policy
    torch.save(policy.state_dict(), "policy_sft_pretrained.pth")
    print("Saved pre-trained policy!")
```

---

### Step 2: Modify TRPO to Load Pre-trained Policy

```python
# In trpo_torch.py, add option to load pre-trained:

if args.load_sft_pretrain:
    print("Loading SFT pre-trained policy...")
    policy.load_state_dict(torch.load("policy_sft_pretrained.pth"))
    print("Starting RL fine-tuning from pre-trained policy")

# Then continue with normal TRPO training
agent = TRPO(env, policy, ...)
```

---

## Expected Results

### Pure RL (Current)
```
Iteration 0:    reward = 3.5  (random)
Iteration 50:   reward = 3.8  (slight improvement)
Iteration 200:  reward = 5.0  (learning slowly)
Iteration 1000: reward = 7.5  (good after long training)
```

### SFT Pre-training + RL Fine-tuning
```
SFT training:   (30 minutes)
Iteration 0:    reward = 6.5  (already good!)
Iteration 50:   reward = 7.8  (refining physics)
Iteration 200:  reward = 8.5  (excellent, robust)
```

---

## Addressing Distribution Shift (Advanced)

### The Problem
- **Training**: Policy sees perfect mocap states
- **Testing**: Policy sees its own (imperfect) predictions
- **Result**: Performance degrades over time

### Solution: DAgger (Dataset Aggregation)

```python
# Iterative approach to fix distribution shift
def train_with_dagger(policy, mocap_data, env, num_rounds=10):
    dataset = extract_initial_dataset(mocap_data)
    
    for round in range(num_rounds):
        # 1. Train on current dataset (SFT)
        train_sft(policy, dataset)
        
        # 2. Roll out policy in environment
        trajectories = collect_trajectories(policy, env)
        
        # 3. Ask expert (mocap) what to do at policy's states
        for traj in trajectories:
            for obs in traj.observations:
                # Find nearest mocap frame
                mocap_frame = find_nearest(obs, mocap_data)
                expert_action = compute_action(mocap_frame)
                
                # Add to dataset
                dataset.append((obs, expert_action))
        
        print(f"Round {round}: Dataset size = {len(dataset)}")
    
    return policy
```

---

## Conclusion

**Should you use SFT?** **YES!**

**Recommended approach:**
1. **Start with SFT** (1-2 hours training)
   - Get policy that roughly imitates mocap
   - Much faster than pure RL
   
2. **Fine-tune with RL** (a few hours)
   - Handle physics imperfections
   - Learn recovery behaviors
   - Improve robustness

3. **(Optional) Use DAgger** for best quality
   - Fixes distribution shift
   - More complex to implement

**Implementation priority:**
1. ✅ **Do first**: Simple SFT (Option A above)
2. ✅ **Then**: Hybrid SFT + RL
3. ⏭️ **Advanced**: DAgger if needed

This approach is used in many state-of-the-art motion imitation systems and will likely give you better results faster than pure RL!

Would you like me to implement the SFT training script for your codebase?
