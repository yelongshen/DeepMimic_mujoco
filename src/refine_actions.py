#!/usr/bin/env python3
"""
Iterative Action Refinement for Better Mocap Reproduction

This module provides methods to refine PD-control actions to minimize
trajectory tracking error through iterative optimization.
"""

import os
import sys
import numpy as np
import torch
from tqdm import tqdm
from copy import deepcopy

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from dp_env_v3 import DPEnv
from deepmimic_mujoco.mocap_v2 import MocapDM


class ActionRefiner:
    """
    Iteratively refine actions to minimize trajectory tracking error
    """
    
    def __init__(self, env, mocap_path):
        self.env = env
        self.mocap = MocapDM()
        self.mocap.load_mocap(mocap_path)
        
        print(f"Loaded mocap: {len(self.mocap.data_config)} frames")
    
    def compute_initial_actions_pd(self, kp=1.0, kd=0.1):
        """Compute initial actions using PD control"""
        print("\nComputing initial PD control actions...")
        actions = []
        
        for i in tqdm(range(len(self.mocap.data_config) - 1)):
            qpos_current = self.mocap.data_config[i]
            qvel_current = self.mocap.data_vel[i]
            qpos_target = self.mocap.data_config[i + 1]
            
            # PD control
            current_joints = qpos_current[7:]
            target_joints = qpos_target[7:]
            joint_vels = qvel_current[6:]
            
            action = kp * (target_joints - current_joints) - kd * joint_vels
            action = np.clip(action, -1.0, 1.0)
            actions.append(action)
        
        return np.array(actions)
    
    def evaluate_actions(self, actions):
        """
        Simulate actions and compute tracking error
        
        Returns:
            dict with errors and simulated trajectory
        """
        qpos_sim = [self.mocap.data_config[0].copy()]
        qvel_sim = [self.mocap.data_vel[0].copy()]
        
        # Set initial state
        self.env.set_state(self.mocap.data_config[0], self.mocap.data_vel[0])
        
        for action in actions:
            obs, reward, done, info = self.env.step(action)
            qpos_sim.append(self.env.sim.data.qpos.copy())
            qvel_sim.append(self.env.sim.data.qvel.copy())
        
        qpos_sim = np.array(qpos_sim)
        qpos_mocap = self.mocap.data_config[:len(qpos_sim)]
        
        # Compute errors
        joint_errors = np.abs(qpos_sim[:, 7:] - qpos_mocap[:, 7:])
        root_errors = np.linalg.norm(qpos_sim[:, :3] - qpos_mocap[:, :3], axis=1)
        
        return {
            'qpos_sim': qpos_sim,
            'qvel_sim': qvel_sim,
            'joint_errors': joint_errors,
            'root_errors': root_errors,
            'mean_joint_error': joint_errors.mean(),
            'max_joint_error': joint_errors.max(),
            'mean_root_error': root_errors.mean(),
            'max_root_error': root_errors.max(),
        }
    
    # ============================================================
    # Method 1: Gradient-Based Refinement (BEST)
    # ============================================================
    
    def refine_actions_gradient(self, actions_init, num_iterations=10, 
                               learning_rate=0.01, verbose=True):
        """
        Refine actions using gradient descent to minimize tracking error
        
        This is the most principled approach - directly optimize actions
        to minimize the difference between simulated and mocap trajectories.
        
        Args:
            actions_init: Initial actions [N, 28]
            num_iterations: Number of refinement iterations
            learning_rate: Step size for gradient descent
            
        Returns:
            refined_actions: Optimized actions [N, 28]
        """
        print(f"\n{'='*70}")
        print(f"Method 1: Gradient-Based Refinement")
        print(f"{'='*70}")
        
        # Convert to torch tensor
        actions = torch.tensor(actions_init, dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([actions], lr=learning_rate)
        
        initial_eval = self.evaluate_actions(actions_init)
        print(f"\nInitial errors:")
        print(f"  Mean joint error: {initial_eval['mean_joint_error']:.6f} rad")
        print(f"  Mean root error:  {initial_eval['mean_root_error']:.6f} m")
        
        print(f"\nRefining actions ({num_iterations} iterations)...")
        
        for iter in range(num_iterations):
            optimizer.zero_grad()
            
            # Simulate with current actions
            self.env.set_state(self.mocap.data_config[0], self.mocap.data_vel[0])
            
            total_loss = 0.0
            qpos_errors = []
            
            for i, action in enumerate(actions):
                # Clip actions to valid range
                action_clipped = torch.clamp(action, -1.0, 1.0)
                
                # Apply action (convert to numpy)
                obs, reward, done, info = self.env.step(action_clipped.detach().numpy())
                
                # Get resulting state
                qpos_actual = torch.tensor(self.env.sim.data.qpos.copy(), dtype=torch.float32)
                qpos_target = torch.tensor(self.mocap.data_config[i + 1], dtype=torch.float32)
                
                # Compute loss (MSE on joint positions)
                joint_error = torch.sum((qpos_actual[7:] - qpos_target[7:]) ** 2)
                root_error = torch.sum((qpos_actual[:3] - qpos_target[:3]) ** 2)
                
                loss = joint_error + 0.1 * root_error  # Weight root less
                total_loss += loss
                
                qpos_errors.append(joint_error.item())
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Clip actions after update
            with torch.no_grad():
                actions.clamp_(-1.0, 1.0)
            
            if verbose and (iter % 2 == 0 or iter == num_iterations - 1):
                mean_error = np.mean(qpos_errors)
                print(f"  Iter {iter:2d}: loss={total_loss.item():.4f}, "
                      f"mean_error={mean_error:.6f}")
        
        # Final evaluation
        actions_refined = actions.detach().numpy()
        final_eval = self.evaluate_actions(actions_refined)
        
        print(f"\nFinal errors:")
        print(f"  Mean joint error: {final_eval['mean_joint_error']:.6f} rad "
              f"(improved by {initial_eval['mean_joint_error'] - final_eval['mean_joint_error']:.6f})")
        print(f"  Mean root error:  {final_eval['mean_root_error']:.6f} m "
              f"(improved by {initial_eval['mean_root_error'] - final_eval['mean_root_error']:.6f})")
        
        return actions_refined
    
    # ============================================================
    # Method 2: Error Feedback Correction (SIMPLE & FAST)
    # ============================================================
    
    def refine_actions_feedback(self, actions_init, num_iterations=5, 
                               alpha=0.5, verbose=True):
        """
        Refine actions by adding feedback based on tracking error
        
        Intuition: If we undershoot the target, increase the action.
        If we overshoot, decrease the action.
        
        This is simpler than gradient descent but less principled.
        
        Args:
            actions_init: Initial actions [N, 28]
            num_iterations: Number of refinement passes
            alpha: Feedback gain (0-1)
            
        Returns:
            refined_actions: Corrected actions [N, 28]
        """
        print(f"\n{'='*70}")
        print(f"Method 2: Error Feedback Correction")
        print(f"{'='*70}")
        
        actions = actions_init.copy()
        
        initial_eval = self.evaluate_actions(actions)
        print(f"\nInitial errors:")
        print(f"  Mean joint error: {initial_eval['mean_joint_error']:.6f} rad")
        print(f"  Mean root error:  {initial_eval['mean_root_error']:.6f} m")
        
        print(f"\nRefining actions ({num_iterations} iterations)...")
        
        for iter in range(num_iterations):
            # Simulate with current actions
            self.env.set_state(self.mocap.data_config[0], self.mocap.data_vel[0])
            
            qpos_errors = []
            
            for i in range(len(actions)):
                # Apply action
                obs, reward, done, info = self.env.step(actions[i])
                
                # Compute error
                qpos_actual = self.env.sim.data.qpos.copy()
                qpos_target = self.mocap.data_config[i + 1]
                
                # Joint error (what we care about)
                joint_error = qpos_target[7:] - qpos_actual[7:]  # [28]
                qpos_errors.append(np.abs(joint_error).mean())
                
                # Correct action based on error
                # If we undershot, add positive correction
                # If we overshot, add negative correction
                correction = alpha * joint_error
                actions[i] = np.clip(actions[i] + correction, -1.0, 1.0)
            
            mean_error = np.mean(qpos_errors)
            if verbose:
                print(f"  Iter {iter:2d}: mean_error={mean_error:.6f}")
        
        # Final evaluation
        final_eval = self.evaluate_actions(actions)
        
        print(f"\nFinal errors:")
        print(f"  Mean joint error: {final_eval['mean_joint_error']:.6f} rad "
              f"(improved by {initial_eval['mean_joint_error'] - final_eval['mean_joint_error']:.6f})")
        print(f"  Mean root error:  {final_eval['mean_root_error']:.6f} m "
              f"(improved by {initial_eval['mean_root_error'] - final_eval['mean_root_error']:.6f})")
        
        return actions
    
    # ============================================================
    # Method 3: Model Predictive Control (MPC) Style
    # ============================================================
    
    def refine_actions_mpc(self, actions_init, horizon=5, num_samples=20, 
                          verbose=True):
        """
        Refine actions using MPC-style sampling and selection
        
        For each step:
        1. Sample multiple action variations
        2. Simulate short horizon
        3. Select action that minimizes future error
        
        This is computationally expensive but can work well.
        
        Args:
            actions_init: Initial actions [N, 28]
            horizon: How many steps to look ahead
            num_samples: Number of action variations to try
            
        Returns:
            refined_actions: Optimized actions [N, 28]
        """
        print(f"\n{'='*70}")
        print(f"Method 3: MPC-Style Sampling")
        print(f"{'='*70}")
        
        actions = actions_init.copy()
        
        initial_eval = self.evaluate_actions(actions)
        print(f"\nInitial errors:")
        print(f"  Mean joint error: {initial_eval['mean_joint_error']:.6f} rad")
        
        print(f"\nRefining actions (sampling {num_samples} variations per step)...")
        
        # Refine each action
        for i in tqdm(range(len(actions))):
            best_action = actions[i]
            best_error = float('inf')
            
            # Try different action variations
            for _ in range(num_samples):
                # Sample action variation
                noise = np.random.normal(0, 0.1, size=actions[i].shape)
                action_candidate = np.clip(actions[i] + noise, -1.0, 1.0)
                
                # Simulate from current mocap state
                self.env.set_state(self.mocap.data_config[i], self.mocap.data_vel[i])
                obs, reward, done, info = self.env.step(action_candidate)
                
                # Evaluate error
                qpos_actual = self.env.sim.data.qpos
                qpos_target = self.mocap.data_config[i + 1]
                error = np.abs(qpos_actual[7:] - qpos_target[7:]).mean()
                
                # Keep best action
                if error < best_error:
                    best_error = error
                    best_action = action_candidate
            
            actions[i] = best_action
        
        # Final evaluation
        final_eval = self.evaluate_actions(actions)
        
        print(f"\nFinal errors:")
        print(f"  Mean joint error: {final_eval['mean_joint_error']:.6f} rad "
              f"(improved by {initial_eval['mean_joint_error'] - final_eval['mean_joint_error']:.6f})")
        
        return actions
    
    # ============================================================
    # Method 4: Inverse Dynamics (MOST ACCURATE)
    # ============================================================
    
    def refine_actions_inverse_dynamics(self, verbose=True):
        """
        Compute actions using inverse dynamics (most accurate)
        
        This computes the EXACT torques needed to produce mocap accelerations.
        Requires computing accelerations from mocap velocities.
        
        Returns:
            actions: Physically accurate actions [N, 28]
        """
        print(f"\n{'='*70}")
        print(f"Method 4: Inverse Dynamics (Ground Truth)")
        print(f"{'='*70}")
        
        try:
            import mujoco
        except ImportError:
            print("❌ Error: mujoco package not found!")
            print("Install with: pip install mujoco")
            return None
        
        print("\nComputing accelerations from velocities...")
        dt = self.mocap.dt
        
        # Compute accelerations via finite differences
        qaccs = []
        for i in range(len(self.mocap.data_vel) - 1):
            qvel_curr = self.mocap.data_vel[i]
            qvel_next = self.mocap.data_vel[i + 1]
            qacc = (qvel_next - qvel_curr) / dt
            qaccs.append(qacc)
        
        print(f"Computing inverse dynamics for {len(qaccs)} frames...")
        
        actions = []
        for i in tqdm(range(len(qaccs))):
            # Set MuJoCo state
            qpos = self.mocap.data_config[i]
            qvel = self.mocap.data_vel[i]
            qacc = qaccs[i]
            
            self.env.sim.data.qpos[:] = qpos
            self.env.sim.data.qvel[:] = qvel
            self.env.sim.data.qacc[:] = qacc
            
            # Forward kinematics
            self.env.sim.forward()
            
            # Inverse dynamics
            mujoco.mj_inverse(self.env.sim.model, self.env.sim.data)
            
            # Extract torques (skip free joint)
            torques = self.env.sim.data.qfrc_inverse[6:34]  # [28]
            
            # Convert to actions (normalize by actuator gear)
            actuator_gear = self.env.sim.model.actuator_gear[:, 0]
            action = torques / (actuator_gear + 1e-6)
            action = np.clip(action, -1.0, 1.0)
            
            actions.append(action)
        
        actions = np.array(actions)
        
        # Evaluate
        eval_results = self.evaluate_actions(actions)
        
        print(f"\nInverse dynamics errors:")
        print(f"  Mean joint error: {eval_results['mean_joint_error']:.6f} rad")
        print(f"  Mean root error:  {eval_results['mean_root_error']:.6f} m")
        
        return actions


def compare_methods(mocap_path='deepmimic_mujoco/motions/humanoid3d_dance_a.txt'):
    """Compare all refinement methods"""
    print("="*70)
    print("Comparing Action Refinement Methods")
    print("="*70)
    
    env = DPEnv()
    refiner = ActionRefiner(env, mocap_path)
    
    # Compute initial PD actions
    actions_pd = refiner.compute_initial_actions_pd(kp=1.0, kd=0.1)
    
    results = {}
    
    # Method 1: Gradient-based
    actions_grad = refiner.refine_actions_gradient(
        actions_pd, num_iterations=10, learning_rate=0.01
    )
    results['gradient'] = refiner.evaluate_actions(actions_grad)
    
    # Method 2: Feedback correction
    actions_feedback = refiner.refine_actions_feedback(
        actions_pd, num_iterations=5, alpha=0.5
    )
    results['feedback'] = refiner.evaluate_actions(actions_feedback)
    
    # Method 3: MPC sampling
    actions_mpc = refiner.refine_actions_mpc(
        actions_pd, horizon=5, num_samples=20
    )
    results['mpc'] = refiner.evaluate_actions(actions_mpc)
    
    # Method 4: Inverse dynamics (if available)
    try:
        actions_invdyn = refiner.refine_actions_inverse_dynamics()
        if actions_invdyn is not None:
            results['inverse_dynamics'] = refiner.evaluate_actions(actions_invdyn)
    except Exception as e:
        print(f"\n⚠️  Inverse dynamics failed: {e}")
    
    # Print comparison
    print(f"\n{'='*70}")
    print("Method Comparison")
    print(f"{'='*70}")
    print(f"{'Method':<20} {'Mean Joint Error':>15} {'Mean Root Error':>15}")
    print(f"{'-'*70}")
    
    for method, result in results.items():
        print(f"{method:<20} {result['mean_joint_error']:>15.6f} {result['mean_root_error']:>15.6f}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mocap', type=str,
                       default='deepmimic_mujoco/motions/humanoid3d_dance_a.txt')
    parser.add_argument('--method', type=str, default='all',
                       choices=['gradient', 'feedback', 'mpc', 'invdyn', 'all'])
    args = parser.parse_args()
    
    if args.method == 'all':
        compare_methods(args.mocap)
    else:
        env = DPEnv()
        refiner = ActionRefiner(env, args.mocap)
        actions_pd = refiner.compute_initial_actions_pd()
        
        if args.method == 'gradient':
            refiner.refine_actions_gradient(actions_pd)
        elif args.method == 'feedback':
            refiner.refine_actions_feedback(actions_pd)
        elif args.method == 'mpc':
            refiner.refine_actions_mpc(actions_pd)
        elif args.method == 'invdyn':
            refiner.refine_actions_inverse_dynamics()
