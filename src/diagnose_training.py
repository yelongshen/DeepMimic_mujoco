#!/usr/bin/env python
"""
Diagnostic script to check why TRPO training isn't improving
"""

import numpy as np
import sys
sys.path.append('env')
from dp_env_v3 import DPEnv

def test_reward_function():
    """Test the reward function to see typical values"""
    print("="*60)
    print("Testing Reward Function")
    print("="*60)
    
    # Create environment
    env = DPEnv()
    
    # Run random policy for a few episodes
    total_rewards = []
    episode_lengths = []
    min_rewards = []
    max_rewards = []
    
    for ep in range(10):
        ob = env.reset()
        ob = env.reset_model_init()
        
        ep_rewards = []
        done = False
        step = 0
        max_steps = 200
        
        while not done and step < max_steps:
            # Random action
            action = env.action_space.sample()
            ob, reward, done, info = env.step(action)
            ep_rewards.append(reward)
            step += 1
        
        total_rewards.append(np.sum(ep_rewards))
        episode_lengths.append(len(ep_rewards))
        min_rewards.append(np.min(ep_rewards))
        max_rewards.append(np.max(ep_rewards))
        
        print(f"Episode {ep+1}: Length={len(ep_rewards)}, "
              f"Total={np.sum(ep_rewards):.4f}, "
              f"Mean={np.mean(ep_rewards):.6f}, "
              f"Min={np.min(ep_rewards):.6f}, "
              f"Max={np.max(ep_rewards):.6f}")
    
    print()
    print("Summary Statistics:")
    print(f"  Avg episode length: {np.mean(episode_lengths):.2f}")
    print(f"  Avg total reward: {np.mean(total_rewards):.4f}")
    avg_step_rewards = [total_rewards[i]/episode_lengths[i] for i in range(len(total_rewards))]
    print(f"  Avg step reward: {np.mean(avg_step_rewards):.6f}")
    print(f"  Min reward seen: {np.min(min_rewards):.6f}")
    print(f"  Max reward seen: {np.max(max_rewards):.6f}")
    print()
    
    # Test what perfect imitation would give
    print("="*60)
    print("Testing Perfect Imitation Reward")
    print("="*60)
    
    ob = env.reset()
    ob = env.reset_model_init()
    
    # Get current mocap target
    target_config = env.mocap.data_config[env.idx_curr][7:]
    curr_config = env.get_joint_configs()
    
    # Calculate error
    err = env.calc_config_errs(curr_config, target_config)
    reward = np.exp(-err)
    
    print(f"Initial state:")
    print(f"  Joint config error: {err:.4f}")
    print(f"  Reward: {reward:.6f}")
    print(f"  Config shape: {curr_config.shape}, Target shape: {target_config.shape}")
    print()
    
    # What if error was small?
    print("Theoretical rewards for different error levels:")
    for err_test in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        rew_test = np.exp(-err_test)
        print(f"  Error={err_test:5.1f} → Reward={rew_test:.6f}")
    print()
    
    # Check if the environment is actually using the reward function
    print("="*60)
    print("Verifying Reward Calculation in Step")
    print("="*60)
    
    ob = env.reset()
    ob = env.reset_model_init()
    
    for i in range(5):
        action = env.action_space.sample()
        ob, reward, done, info = env.step(action)
        
        # Manually calculate what reward should be
        expected_reward = env.calc_config_reward()
        
        print(f"Step {i+1}: reward={reward:.6f}, expected={expected_reward:.6f}, match={'✓' if abs(reward - expected_reward) < 1e-6 else '✗ MISMATCH!'}")
    
    print()
    env.close()

if __name__ == "__main__":
    test_reward_function()
