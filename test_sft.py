#!/usr/bin/env python3
"""
Quick test to verify SFT implementation works
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("Testing SFT implementation...")
print("=" * 60)

# Test 1: Import modules
print("\n1. Testing imports...")
try:
    from dp_env_v3 import DPEnv
    from mlp_policy_torch import MlpPolicy
    print("   ✓ Environment and policy imported successfully")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Create environment
print("\n2. Testing environment creation...")
try:
    env = DPEnv()
    print(f"   ✓ Environment created")
    print(f"     - Observation space: {env.observation_space.shape}")
    print(f"     - Action space: {env.action_space.shape}")
    print(f"     - Mocap frames: {len(env.mocap.data)}")
except Exception as e:
    print(f"   ✗ Environment creation failed: {e}")
    sys.exit(1)

# Test 3: Create policy
print("\n3. Testing policy creation...")
try:
    policy = MlpPolicy(env.observation_space, env.action_space)
    print(f"   ✓ Policy created")
    
    # Count parameters
    num_params = sum(p.numel() for p in policy.parameters())
    print(f"     - Total parameters: {num_params:,}")
except Exception as e:
    print(f"   ✗ Policy creation failed: {e}")
    sys.exit(1)

# Test 4: Test PD control computation
print("\n4. Testing PD control action computation...")
try:
    qpos_current = env.mocap.data_config[0]
    qvel_current = env.mocap.data_vel[0]
    qpos_target = env.mocap.data_config[1]
    
    # Compute action
    kp, kd = 1.0, 0.1
    current_joints = qpos_current[7:]
    target_joints = qpos_target[7:]
    joint_vels = qvel_current[6:]
    
    action = kp * (target_joints - current_joints) - kd * joint_vels
    action = np.clip(action, -1.0, 1.0)
    
    print(f"   ✓ PD control computed successfully")
    print(f"     - Action shape: {action.shape}")
    print(f"     - Action range: [{action.min():.3f}, {action.max():.3f}]")
    print(f"     - Action mean: {action.mean():.3f}")
except Exception as e:
    print(f"   ✗ PD control computation failed: {e}")
    sys.exit(1)

# Test 5: Test observation extraction
print("\n5. Testing observation extraction...")
try:
    env.set_state(qpos_current, qvel_current)
    obs = env._get_obs()
    
    print(f"   ✓ Observation extracted successfully")
    print(f"     - Observation shape: {obs.shape}")
    print(f"     - Expected shape: {env.observation_space.shape}")
    assert obs.shape == env.observation_space.shape, "Shape mismatch!"
except Exception as e:
    print(f"   ✗ Observation extraction failed: {e}")
    sys.exit(1)

# Test 6: Test policy forward pass
print("\n6. Testing policy forward pass...")
try:
    import torch
    # Note: policy.act() expects observation WITHOUT batch dimension
    obs_tensor = torch.tensor(obs, dtype=torch.float32)
    
    action, value = policy.act(obs_tensor, stochastic=False)
    # action is already a numpy array with shape (ac_dim,) from policy.act()
    
    print(f"   ✓ Policy forward pass successful")
    print(f"     - Output action shape: {action.shape}")
    print(f"     - Expected shape: {env.action_space.shape}")
    assert action.shape == env.action_space.shape, "Shape mismatch!"
except Exception as e:
    print(f"   ✗ Policy forward pass failed: {e}")
    sys.exit(1)

# Test 7: Test dataset extraction (small sample)
print("\n7. Testing dataset extraction (10 samples)...")
try:
    dataset = []
    for i in range(10):
        qpos = env.mocap.data_config[i]
        qvel = env.mocap.data_vel[i]
        qpos_next = env.mocap.data_config[i + 1]
        
        env.set_state(qpos, qvel)
        obs = env._get_obs()
        
        # Compute action
        current_joints = qpos[7:]
        target_joints = qpos_next[7:]
        joint_vels = qvel[6:]
        action = kp * (target_joints - current_joints) - kd * joint_vels
        action = np.clip(action, -1.0, 1.0)
        
        dataset.append((obs, action))
    
    print(f"   ✓ Dataset extraction successful")
    print(f"     - Dataset size: {len(dataset)} samples")
    print(f"     - Observation shape: {dataset[0][0].shape}")
    print(f"     - Action shape: {dataset[0][1].shape}")
except Exception as e:
    print(f"   ✗ Dataset extraction failed: {e}")
    sys.exit(1)

# Test 8: Test training loop (1 batch)
print("\n8. Testing training loop (1 batch)...")
try:
    import torch
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    
    obs_batch = torch.tensor([x[0] for x in dataset], dtype=torch.float32)
    action_batch = torch.tensor([x[1] for x in dataset], dtype=torch.float32)
    
    # Forward pass
    obs_normalized = (obs_batch - policy.ob_rms.mean) / policy.ob_rms.std
    pol_output = policy.pol_net(obs_normalized)
    predicted_actions = policy.pol_mean(pol_output)
    
    # Loss
    loss = torch.mean((predicted_actions - action_batch) ** 2)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"   ✓ Training loop successful")
    print(f"     - Batch size: {obs_batch.shape[0]}")
    print(f"     - Loss: {loss.item():.6f}")
except Exception as e:
    print(f"   ✗ Training loop failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("=" * 60)
print("\nYou can now run:")
print("  ./run_sft_train.sh")
print("\nOr:")
print("  cd src && python train_sft.py")
print()
