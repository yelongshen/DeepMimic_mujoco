#!/usr/bin/env python3

from mlp_policy_torch import MlpPolicy
import numpy as np
import torch

try:
    import gymnasium as gym
except ImportError:
    import gym

def get_flat_params(model, param_list=None):
    """Get flattened model parameters"""
    if param_list is None:
        param_list = list(model.parameters())
    
    return torch.cat([param.data.reshape(-1) for param in param_list])

# Create spaces
ob_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(197,), dtype=np.float32)
ac_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(36,), dtype=np.float32)

# Create a policy
policy = MlpPolicy(ob_space, ac_space)

print("All parameter names in MlpPolicy:")
for name, p in policy.named_parameters():
    print(f"  {name}: {p.shape}")
    
print("\nPolicy parameters (with 'pol' filter):")
policy_params = [p for name, p in policy.named_parameters() 
                if 'pol' in name or 'logstd' in name]
print(f"Found {len(policy_params)} parameters")

print("\nTest get_flat_params:")
flat = get_flat_params(policy, policy_params)
print(f"Flat params shape: {flat.shape}")
print(f"Flat params norm: {torch.norm(flat).item()}")
