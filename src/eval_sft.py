#!/usr/bin/env python3
"""
Evaluate a trained SFT model
"""

import os
import sys
import argparse
import numpy as np
import torch

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from mlp_policy_torch import MlpPolicy
from dp_env_v3 import DPEnv


def evaluate_policy(policy, env, num_episodes=10, max_steps=500):
    """Evaluate policy in environment"""
    print(f"Evaluating policy for {num_episodes} episodes...")
    
    policy.eval()
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # Get action from policy
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            
            with torch.no_grad():
                action, _ = policy.act(obs_tensor, stochastic=False)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        print(f"  Episode {episode+1}: Reward = {episode_reward:.2f}")
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    print(f"\nEvaluation Results:")
    print(f"  Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"  Min Reward:  {np.min(episode_rewards):.2f}")
    print(f"  Max Reward:  {np.max(episode_rewards):.2f}")
    
    return mean_reward, std_reward


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained SFT policy')
    parser.add_argument('--model_path', type=str, default='policy_sft_pretrained.pth',
                       help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to evaluate')
    parser.add_argument('--max_steps', type=int, default=500,
                       help='Maximum steps per episode')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Evaluating Trained SFT Policy")
    print("=" * 70)
    print(f"\nModel: {args.model_path}")
    print(f"Episodes: {args.episodes}")
    print()
    
    # Create environment
    print("Creating environment...")
    env = DPEnv()
    
    # Create policy
    print("Creating policy...")
    policy = MlpPolicy(
        ob_space=env.observation_space,
        ac_space=env.action_space
    )
    
    # Load trained model
    print(f"Loading model from {args.model_path}...")
    policy.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    print("✓ Model loaded successfully\n")
    
    # Evaluate
    evaluate_policy(policy, env, args.episodes, args.max_steps)
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
