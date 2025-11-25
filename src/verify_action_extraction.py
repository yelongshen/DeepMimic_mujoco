#!/usr/bin/env python3
"""
Verify that extracted actions correctly reproduce mocap motion

This script tests if applying the PD-control-computed actions
actually moves the character along the mocap trajectory.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from dp_env_v3 import DPEnv
from deepmimic_mujoco.mocap_v2 import MocapDM
from train_sft import MotionImitationSFT


class ActionVerifier:
    """Verify that extracted actions reproduce mocap motion"""
    
    def __init__(self, env, mocap_path):
        self.env = env
        self.mocap = MocapDM()
        self.mocap.load_mocap(mocap_path)
        
        # Create SFT instance for action computation
        from mlp_policy_torch import MlpPolicy
        policy = MlpPolicy(env.observation_space, env.action_space)
        self.sft = MotionImitationSFT(env, policy, mocap_path)
    
    def test_single_step(self, frame_idx=0, verbose=True):
        """
        Test if action at frame_idx moves character toward frame_idx+1
        
        Returns:
            dict with metrics
        """
        if frame_idx >= len(self.mocap.data_config) - 1:
            raise ValueError(f"frame_idx must be < {len(self.mocap.data_config) - 1}")
        
        # Get mocap states
        qpos_start = self.mocap.data_config[frame_idx].copy()
        qvel_start = self.mocap.data_vel[frame_idx].copy()
        qpos_target = self.mocap.data_config[frame_idx + 1].copy()
        
        # Compute action using PD control
        action = self.sft.compute_action_pd_control(
            qpos_start, qvel_start, qpos_target
        )
        
        # Set environment to start state and apply action
        self.env.set_state(qpos_start, qvel_start)
        obs, reward, done, info = self.env.step(action)
        
        # Get resulting state
        qpos_actual = self.env.sim.data.qpos.copy()
        qvel_actual = self.env.sim.data.qvel.copy()
        
        # Compute errors
        # Joint positions (skip root position/orientation)
        joints_target = qpos_target[7:]  # [28]
        joints_actual = qpos_actual[7:]  # [28]
        joints_start = qpos_start[7:]    # [28]
        
        position_error = np.abs(joints_actual - joints_target)
        position_improvement = np.abs(joints_start - joints_target) - position_error
        
        # Root position error (first 3 elements)
        root_pos_error = np.linalg.norm(qpos_actual[:3] - qpos_target[:3])
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Single Step Test (Frame {frame_idx} → {frame_idx + 1})")
            print(f"{'='*70}")
            print(f"\nAction statistics:")
            print(f"  Min:  {action.min():.4f}")
            print(f"  Max:  {action.max():.4f}")
            print(f"  Mean: {action.mean():.4f}")
            print(f"  Std:  {action.std():.4f}")
            
            print(f"\nJoint position errors (28 joints):")
            print(f"  Mean error:        {position_error.mean():.6f} rad")
            print(f"  Max error:         {position_error.max():.6f} rad")
            print(f"  Mean improvement:  {position_improvement.mean():.6f} rad")
            
            print(f"\nRoot position error: {root_pos_error:.6f} m")
            print(f"Reward: {reward:.4f}")
            
            print(f"\nWorst 5 joints:")
            worst_indices = np.argsort(position_error)[-5:][::-1]
            for idx in worst_indices:
                print(f"  Joint {idx:2d}: error={position_error[idx]:.6f} rad "
                      f"(improvement={position_improvement[idx]:+.6f})")
        
        return {
            'frame_idx': frame_idx,
            'action': action,
            'position_error_mean': position_error.mean(),
            'position_error_max': position_error.max(),
            'position_improvement_mean': position_improvement.mean(),
            'root_pos_error': root_pos_error,
            'reward': reward,
        }
    
    def test_trajectory_following(self, num_steps=50, start_frame=0):
        """
        Test if repeatedly applying actions follows the mocap trajectory
        
        This simulates what happens during actual playback
        """
        print(f"\n{'='*70}")
        print(f"Trajectory Following Test ({num_steps} steps)")
        print(f"{'='*70}")
        
        # Initialize at mocap start
        qpos_init = self.mocap.data_config[start_frame].copy()
        qvel_init = self.mocap.data_vel[start_frame].copy()
        self.env.set_state(qpos_init, qvel_init)
        
        # Track states
        sim_qpos_history = [qpos_init.copy()]
        mocap_qpos_history = [qpos_init.copy()]
        action_history = []
        reward_history = []
        
        print("Simulating trajectory...")
        for step in tqdm(range(num_steps)):
            if start_frame + step + 1 >= len(self.mocap.data_config):
                print(f"Reached end of mocap at step {step}")
                break
            
            # Get current state
            qpos_current = self.env.sim.data.qpos.copy()
            qvel_current = self.env.sim.data.qvel.copy()
            
            # Get target from mocap
            qpos_target = self.mocap.data_config[start_frame + step + 1]
            
            # Compute action
            action = self.sft.compute_action_pd_control(
                qpos_current, qvel_current, qpos_target
            )
            
            # Apply action
            obs, reward, done, info = self.env.step(action)
            
            # Record
            sim_qpos_history.append(self.env.sim.data.qpos.copy())
            mocap_qpos_history.append(qpos_target.copy())
            action_history.append(action)
            reward_history.append(reward)
        
        # Convert to arrays
        sim_qpos = np.array(sim_qpos_history)
        mocap_qpos = np.array(mocap_qpos_history)
        actions = np.array(action_history)
        rewards = np.array(reward_history)
        
        # Compute errors over time
        # Joint positions (skip root)
        sim_joints = sim_qpos[:, 7:]    # [T, 28]
        mocap_joints = mocap_qpos[:, 7:]  # [T, 28]
        joint_errors = np.abs(sim_joints - mocap_joints)  # [T, 28]
        
        # Root position
        sim_root = sim_qpos[:, :3]      # [T, 3]
        mocap_root = mocap_qpos[:, :3]  # [T, 3]
        root_errors = np.linalg.norm(sim_root - mocap_root, axis=1)  # [T]
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"Trajectory Following Results")
        print(f"{'='*70}")
        print(f"\nJoint tracking (28 joints):")
        print(f"  Mean error:   {joint_errors.mean():.6f} rad")
        print(f"  Max error:    {joint_errors.max():.6f} rad")
        print(f"  Final error:  {joint_errors[-1].mean():.6f} rad")
        
        print(f"\nRoot position tracking:")
        print(f"  Mean error:   {root_errors.mean():.6f} m")
        print(f"  Max error:    {root_errors.max():.6f} m")
        print(f"  Final error:  {root_errors[-1]:.6f} m")
        
        print(f"\nRewards:")
        print(f"  Mean:  {rewards.mean():.4f}")
        print(f"  Min:   {rewards.min():.4f}")
        print(f"  Max:   {rewards.max():.4f}")
        
        print(f"\nActions:")
        print(f"  Mean abs: {np.abs(actions).mean():.4f}")
        print(f"  Max abs:  {np.abs(actions).max():.4f}")
        
        # Plot results
        self._plot_trajectory_results(
            joint_errors, root_errors, rewards, actions
        )
        
        return {
            'sim_qpos': sim_qpos,
            'mocap_qpos': mocap_qpos,
            'joint_errors': joint_errors,
            'root_errors': root_errors,
            'rewards': rewards,
            'actions': actions,
        }
    
    def _plot_trajectory_results(self, joint_errors, root_errors, rewards, actions):
        """Plot trajectory following results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Joint tracking errors over time
        ax = axes[0, 0]
        ax.plot(joint_errors.mean(axis=1), label='Mean joint error', linewidth=2)
        ax.fill_between(
            range(len(joint_errors)),
            joint_errors.min(axis=1),
            joint_errors.max(axis=1),
            alpha=0.3,
            label='Min-Max range'
        )
        ax.set_xlabel('Step')
        ax.set_ylabel('Joint Error (rad)')
        ax.set_title('Joint Tracking Error Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Root position error
        ax = axes[0, 1]
        ax.plot(root_errors, linewidth=2, color='orange')
        ax.set_xlabel('Step')
        ax.set_ylabel('Root Position Error (m)')
        ax.set_title('Root Position Tracking Error')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Rewards over time
        ax = axes[1, 0]
        ax.plot(rewards, linewidth=2, color='green')
        ax.axhline(y=rewards.mean(), color='red', linestyle='--', 
                   label=f'Mean: {rewards.mean():.3f}')
        ax.set_xlabel('Step')
        ax.set_ylabel('Reward')
        ax.set_title('Reward Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Action magnitudes
        ax = axes[1, 1]
        action_magnitudes = np.abs(actions)
        ax.plot(action_magnitudes.mean(axis=1), label='Mean |action|', linewidth=2)
        ax.fill_between(
            range(len(actions)),
            action_magnitudes.min(axis=1),
            action_magnitudes.max(axis=1),
            alpha=0.3,
            label='Min-Max range'
        )
        ax.set_xlabel('Step')
        ax.set_ylabel('|Action|')
        ax.set_title('Action Magnitudes Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        save_path = 'action_verification_results.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Plot saved to: {save_path}")
        plt.close()
    
    def test_action_consistency(self, num_samples=100):
        """
        Test if actions are consistent and reasonable
        """
        print(f"\n{'='*70}")
        print(f"Action Consistency Test ({num_samples} samples)")
        print(f"{'='*70}")
        
        actions = []
        for i in tqdm(range(min(num_samples, len(self.mocap.data_config) - 1))):
            qpos_current = self.mocap.data_config[i]
            qvel_current = self.mocap.data_vel[i]
            qpos_target = self.mocap.data_config[i + 1]
            
            action = self.sft.compute_action_pd_control(
                qpos_current, qvel_current, qpos_target
            )
            actions.append(action)
        
        actions = np.array(actions)  # [N, 28]
        
        print(f"\nAction statistics across {len(actions)} samples:")
        print(f"  Shape: {actions.shape}")
        print(f"  Mean:  {actions.mean():.4f}")
        print(f"  Std:   {actions.std():.4f}")
        print(f"  Min:   {actions.min():.4f}")
        print(f"  Max:   {actions.max():.4f}")
        
        # Check if actions are within bounds
        clipped_count = np.sum((actions == -1.0) | (actions == 1.0))
        clipped_percent = 100 * clipped_count / actions.size
        print(f"\n  Clipped values: {clipped_count}/{actions.size} ({clipped_percent:.2f}%)")
        
        if clipped_percent > 50:
            print(f"  ⚠️  Warning: >50% of actions are clipped! Consider:")
            print(f"     - Reducing Kp (currently 1.0)")
            print(f"     - Increasing lookahead_frames")
        elif clipped_percent > 20:
            print(f"  ⚠️  Warning: >20% of actions are clipped")
        else:
            print(f"  ✓ Good: Most actions are within bounds")
        
        # Per-joint statistics
        print(f"\nPer-joint statistics:")
        print(f"  Mean |action| per joint: {np.abs(actions).mean(axis=0).mean():.4f}")
        print(f"  Max |action| per joint:  {np.abs(actions).max(axis=0).mean():.4f}")
        
        # Find most active joints
        joint_activity = np.abs(actions).mean(axis=0)
        most_active = np.argsort(joint_activity)[-5:][::-1]
        print(f"\n  Most active 5 joints:")
        for idx in most_active:
            print(f"    Joint {idx:2d}: mean |action| = {joint_activity[idx]:.4f}")
        
        return {
            'actions': actions,
            'clipped_percent': clipped_percent,
            'joint_activity': joint_activity,
        }


def main():
    parser = argparse.ArgumentParser(description='Verify action extraction from mocap')
    
    parser.add_argument('--mocap', type=str,
                       default='deepmimic_mujoco/motions/humanoid3d_dance_a.txt',
                       help='Path to mocap file')
    parser.add_argument('--test', type=str, default='all',
                       choices=['single', 'trajectory', 'consistency', 'all'],
                       help='Which test to run')
    parser.add_argument('--num_steps', type=int, default=50,
                       help='Number of steps for trajectory test')
    parser.add_argument('--frame_idx', type=int, default=0,
                       help='Frame index for single step test')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Action Extraction Verification")
    print("=" * 70)
    print(f"Mocap file: {args.mocap}")
    print()
    
    # Create environment
    env = DPEnv()
    
    # Create verifier
    verifier = ActionVerifier(env, args.mocap)
    
    # Run tests
    if args.test in ['single', 'all']:
        verifier.test_single_step(frame_idx=args.frame_idx, verbose=True)
    
    if args.test in ['trajectory', 'all']:
        verifier.test_trajectory_following(num_steps=args.num_steps)
    
    if args.test in ['consistency', 'all']:
        verifier.test_action_consistency(num_samples=100)
    
    print("\n" + "=" * 70)
    print("Verification complete!")
    print("=" * 70)
    print("\n📊 Check 'action_verification_results.png' for visualizations")
    print("\n💡 Good results should show:")
    print("  - Joint errors < 0.1 rad")
    print("  - Root errors < 0.1 m")
    print("  - Rewards > 6.0")
    print("  - <20% actions clipped")
    print()


if __name__ == '__main__':
    main()
