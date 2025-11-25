#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) for Motion Imitation
Train policy directly from mocap data using supervised learning
Much faster than pure RL - achieves good results in ~1 hour
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from mlp_policy_torch import MlpPolicy
from dp_env_v3 import DPEnv
from deepmimic_mujoco.mocap_v2 import MocapDM


class MotionImitationSFT:
    """
    Supervised learning from motion capture data
    """
    
    def __init__(self, env, policy, mocap_path, device='cpu'):
        self.env = env
        self.policy = policy.to(device)
        self.device = device
        self.mocap_path = mocap_path
        
        # Load mocap data
        self.mocap = MocapDM()

        print("load mocap path", mocap_path)
        self.mocap.load_mocap(mocap_path)
        
        print(f"Loaded mocap with {len(self.mocap.data)} frames")
        print(f"Mocap dt: {self.mocap.dt}")
        
    def compute_action_pd_control(self, qpos_current, qvel_current, qpos_target):
        """
        Compute action using PD control
        Action tries to move current pose towards target pose
        
        Args:
            qpos_current: Current joint positions [35]
            qvel_current: Current joint velocities [34]
            qpos_target: Target joint positions [35]
            
        Returns:
            action: Control signal [28]
        """
        # PD control gains (tuned for humanoid)
        kp = 1.0  # Proportional gain
        kd = 0.1  # Derivative gain
        
        # Extract joint angles (exclude root)
        current_joints = qpos_current[7:]  # [28]
        target_joints = qpos_target[7:]    # [28]
        joint_vels = qvel_current[6:]      # [28]
        
        # PD control: action = Kp * error - Kd * velocity
        position_error = target_joints - current_joints
        action = kp * position_error - kd * joint_vels
        
        # Clip to action bounds [-1, 1]
        action = np.clip(action, -1.0, 1.0)
        
        return action
    
    def extract_dataset(self, lookahead_frames=1, refine_actions=False, 
                       refine_method='feedback', refine_iterations=3):
        """
        Extract (observation, action) pairs from mocap data
        
        Args:
            lookahead_frames: How many frames ahead to target (1 = next frame)
            refine_actions: Whether to refine actions iteratively
            refine_method: Method for refinement ('feedback', 'gradient', or 'none')
            refine_iterations: Number of refinement iterations
            
        Returns:
            dataset: List of (obs, action) tuples
        """
        print("Extracting dataset from mocap...")
        dataset = []
        
        num_frames = len(self.mocap.data_config)
        
        for i in tqdm(range(num_frames - lookahead_frames)):
            # Current state
            qpos_current = self.mocap.data_config[i]
            qvel_current = self.mocap.data_vel[i]
            
            # Target state (lookahead)
            qpos_target = self.mocap.data_config[i + lookahead_frames]
            
            # Set environment to current state to get observation
            self.env.set_state(qpos_current, qvel_current)
            # x_i
            obs = self.env._get_obs() 
            # Compute action that moves towards target, y_i
            action = self.compute_action_pd_control(
                qpos_current, qvel_current, qpos_target
            )
            
            
            dataset.append((obs, action))
        
        print(f"Extracted {len(dataset)} (observation, action) pairs")
        
        # Optional: Refine actions iteratively
        if refine_actions and refine_method != 'none':
            dataset = self._refine_dataset_actions(
                dataset, method=refine_method, iterations=refine_iterations
            )
        
        # Quick verification of extracted actions
        self._verify_actions(dataset)
        
        return dataset
    
    def _refine_dataset_actions(self, dataset, method='feedback', iterations=3):
        """
        Iteratively refine actions to better match mocap trajectory
        
        Args:
            dataset: List of (obs, action) tuples
            method: Refinement method ('feedback' or 'gradient')
            iterations: Number of refinement iterations
            
        Returns:
            refined_dataset: Dataset with improved actions
        """
        print(f"\n{'='*60}")
        print(f"Refining Actions ({method} method, {iterations} iterations)")
        print(f"{'='*60}")
        
        if method == 'feedback':
            return self._refine_feedback(dataset, iterations, alpha=0.5)
        elif method == 'gradient':
            return self._refine_gradient(dataset, iterations, lr=0.01)
        else:
            print(f"Unknown refinement method: {method}")
            return dataset
    
    def _refine_feedback(self, dataset, iterations, alpha=0.5):
        """Refine actions using error feedback"""
        print("Using error feedback correction...")
        
        refined_dataset = []
        
        for iter_num in range(iterations):
            print(f"\nIteration {iter_num + 1}/{iterations}")
            refined_dataset = []
            errors = []
            
            # Reset to initial mocap state
            self.env.set_state(self.mocap.data_config[0], self.mocap.data_vel[0])
            
            for i in tqdm(range(len(dataset)), desc=f"Iter {iter_num+1}"):
                obs, action = dataset[i]
                
                # Apply current action
                _, _, _, _ = self.env.step(action)
                
                # Get actual and target states
                qpos_actual = self.env.sim.data.qpos.copy()
                qvel_actual = self.env.sim.data.qvel.copy()
                qpos_target = self.mocap.data_config[i + 1]
                
                # Compute error
                joint_error = qpos_target[7:] - qpos_actual[7:]  # [28]
                errors.append(np.abs(joint_error).mean())
                
                # Correct action based on error
                correction = alpha * joint_error
                action_refined = np.clip(action + correction, -1.0, 1.0)
                
                # Get observation from actual state (not mocap)
                obs_refined = self.env._get_obs()
                
                refined_dataset.append((obs_refined, action_refined))
            
            print(f"  Mean error: {np.mean(errors):.6f} rad")
            
            # Use refined dataset for next iteration
            dataset = refined_dataset
        
        print(f"\n✓ Refinement complete!")
        return refined_dataset
    
    def _refine_gradient(self, dataset, iterations, lr=0.01):
        """Refine actions using gradient descent (simplified version)"""
        print("Using gradient-based refinement...")
        print("⚠️  Note: This is a simplified version. For full gradient descent,")
        print("   use the refine_actions.py script.")
        
        # For simplicity, fall back to feedback method
        # Full gradient descent requires differentiable simulator
        return self._refine_feedback(dataset, iterations, alpha=lr*10)
    
    def _verify_actions(self, dataset):
        """Quick sanity check on extracted actions"""
        print("\n" + "="*60)
        print("Action Extraction Verification")
        print("="*60)
        
        # Collect all actions
        actions = np.array([x[1] for x in dataset])  # [N, 28]
        
        print(f"\nAction statistics ({len(actions)} samples):")
        print(f"  Mean:  {actions.mean():.4f}")
        print(f"  Std:   {actions.std():.4f}")
        print(f"  Min:   {actions.min():.4f}")
        print(f"  Max:   {actions.max():.4f}")
        
        # Check clipping
        clipped_count = np.sum((actions == -1.0) | (actions == 1.0))
        clipped_percent = 100 * clipped_count / actions.size
        print(f"  Clipped: {clipped_count}/{actions.size} ({clipped_percent:.1f}%)")
        
        if clipped_percent > 50:
            print(f"  ⚠️  WARNING: >50% actions clipped! Reduce Kp or increase lookahead")
        elif clipped_percent > 20:
            print(f"  ⚠️  Note: {clipped_percent:.1f}% actions clipped")
        else:
            print(f"  ✓ Good: Most actions within bounds")
        
        # Test single step reproduction
        print(f"\n" + "-"*60)
        print("Testing if actions reproduce mocap motion (frame 0→1):")
        print("-"*60)
        
        # Get first frame
        qpos_0 = self.mocap.data_config[0]
        qvel_0 = self.mocap.data_vel[0]
        qpos_1 = self.mocap.data_config[1]
        action_0 = dataset[0][1]  # Extracted action
        
        # Apply action in environment
        self.env.set_state(qpos_0, qvel_0)
        obs, reward, done, info = self.env.step(action_0)
        qpos_actual = self.env.sim.data.qpos.copy()
        
        # Compute error
        joint_error = np.abs(qpos_actual[7:] - qpos_1[7:])  # Joint angles
        root_error = np.linalg.norm(qpos_actual[:3] - qpos_1[:3])  # Root position
        
        print(f"  Joint tracking error:  {joint_error.mean():.6f} rad (mean)")
        print(f"  Root position error:   {root_error:.6f} m")
        print(f"  Reward after 1 step:   {reward:.4f}")
        
        if joint_error.mean() < 0.1 and root_error < 0.1:
            print(f"  ✓ PASS: Actions reproduce mocap motion well!")
        elif joint_error.mean() < 0.2:
            print(f"  ~ OK: Reasonable tracking (may improve with tuning)")
        else:
            print(f"  ⚠️  WARNING: Poor tracking! Check Kp/Kd gains")
        
        print("="*60 + "\n")
    
    def train(self, dataset, num_epochs=100, batch_size=256, learning_rate=1e-3,
              val_split=0.1, save_path=None):
        """
        Train policy with supervised learning
        
        Args:
            dataset: List of (obs, action) tuples
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            val_split: Fraction of data for validation
            save_path: Path to save best model
        """
        # Split into train/val
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        
        np.random.shuffle(dataset)
        train_data = dataset[:train_size]
        val_data = dataset[train_size:]
        
        print(f"\nDataset split:")
        print(f"  Training: {len(train_data)} samples")
        print(f"  Validation: {len(val_data)} samples")
        
        # Update observation statistics for normalization
        print("\nComputing observation statistics for normalization...")
        all_obs = np.array([x[0] for x in train_data])
        self.policy.ob_rms.update(torch.tensor(all_obs, dtype=torch.float32))
        print(f"  Obs mean: {self.policy.ob_rms.mean[:3].numpy()}")  # Print first 3 dims
        print(f"  Obs std:  {self.policy.ob_rms.std[:3].numpy()}")
        
        # Setup optimizer
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        best_val_loss = float('inf')
        
        print(f"\nTraining for {num_epochs} epochs...")
        print("=" * 70)
        
        for epoch in range(num_epochs):
            # Training phase
            self.policy.train()
            np.random.shuffle(train_data)
            
            train_loss = 0.0
            num_batches = 0
            
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i+batch_size]
                
                # Prepare batch
                obs_batch = torch.tensor(
                    [x[0] for x in batch], 
                    dtype=torch.float32, 
                    device=self.device
                )
                action_batch = torch.tensor(
                    [x[1] for x in batch], 
                    dtype=torch.float32, 
                    device=self.device
                )
                
                # Forward pass - get mean action (deterministic)
                # Use policy network to get action distribution
                obs_normalized = (obs_batch - self.policy.ob_rms.mean) / self.policy.ob_rms.std
                pol_output = self.policy.pol_net(obs_normalized)
                
                # Get mean action from distribution
                predicted_actions = self.policy.pol_mean(pol_output)
                
                # Compute loss (MSE)
                loss = torch.mean((predicted_actions - action_batch) ** 2)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
            
            train_loss /= num_batches
            
            # Validation phase
            self.policy.eval()
            val_loss = 0.0
            num_val_batches = 0
            
            with torch.no_grad():
                for i in range(0, len(val_data), batch_size):
                    batch = val_data[i:i+batch_size]
                    
                    obs_batch = torch.tensor(
                        [x[0] for x in batch], 
                        dtype=torch.float32, 
                        device=self.device
                    )
                    action_batch = torch.tensor(
                        [x[1] for x in batch], 
                        dtype=torch.float32, 
                        device=self.device
                    )
                    
                    # Forward pass
                    obs_normalized = (obs_batch - self.policy.ob_rms.mean) / self.policy.ob_rms.std
                    pol_output = self.policy.pol_net(obs_normalized)
                    predicted_actions = self.policy.pol_mean(pol_output)
                    
                    # Compute loss
                    loss = torch.mean((predicted_actions - action_batch) ** 2)
                    val_loss += loss.item()
                    num_val_batches += 1
            
            val_loss /= num_val_batches
            
            # Print progress
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch:3d}/{num_epochs}: "
                      f"Train Loss = {train_loss:.6f}, "
                      f"Val Loss = {val_loss:.6f}")
            
            # Save best model
            if save_path and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.policy.state_dict(), save_path)
                if epoch % 10 == 0:
                    print(f"  → Saved best model (val_loss={val_loss:.6f})")
        
        print("=" * 70)
        print(f"Training complete! Best validation loss: {best_val_loss:.6f}")
        
        if save_path:
            print(f"Best model saved to: {save_path}")
    
    def evaluate_in_environment(self, num_episodes=10, max_steps=500):
        """
        Evaluate trained policy in the actual environment
        
        Args:
            num_episodes: Number of episodes to run
            max_steps: Maximum steps per episode
        """
        print(f"\nEvaluating policy in environment...")
        print(f"Running {num_episodes} episodes...")
        
        self.policy.eval()
        episode_rewards = []
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                # Get action from policy (deterministic)
                # Note: policy.act() expects observation WITHOUT batch dimension
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                
                with torch.no_grad():
                    action, _ = self.policy.act(obs_tensor, stochastic=False)
                
                # action is already a numpy array from policy.act()
                
                # Step environment
                obs, reward, done, info = self.env.step(action)
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
    parser = argparse.ArgumentParser(description='Train policy with supervised learning from mocap')
    
    # Environment args
    parser.add_argument('--mocap', type=str, 
                       default='deepmimic_mujoco/motions/humanoid3d_dance_a.txt',
                       help='Path to mocap file')
    
    # Training args
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--lookahead', type=int, default=1,
                       help='Lookahead frames for target (1 = next frame)')
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='Validation split fraction')
    
    # Action refinement args
    parser.add_argument('--refine_actions', action='store_true',
                       help='Iteratively refine actions to reduce tracking error')
    parser.add_argument('--refine_method', type=str, default='feedback',
                       choices=['feedback', 'gradient', 'none'],
                       help='Method for action refinement')
    parser.add_argument('--refine_iterations', type=int, default=3,
                       help='Number of refinement iterations')
    
    # Evaluation args
    parser.add_argument('--eval_episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--eval_steps', type=int, default=500,
                       help='Maximum steps per evaluation episode')
    
    # Model args
    parser.add_argument('--hidden_size', type=int, default=64,
                       help='Hidden layer size')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of hidden layers')
    
    # Save/load args
    parser.add_argument('--save_path', type=str, default='policy_sft_pretrained.pth',
                       help='Path to save trained model')
    parser.add_argument('--no_eval', action='store_true',
                       help='Skip evaluation in environment')
    
    # Device
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu or cuda)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Supervised Fine-Tuning (SFT) for Motion Imitation")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Mocap file: {args.mocap}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Lookahead frames: {args.lookahead}")
    print(f"  Device: {args.device}")
    print()
    
    # Create environment
    print("Initializing environment...")
    env = DPEnv()
    
    # Create policy
    print("Creating policy network...")
    policy = MlpPolicy(
        ob_space=env.observation_space,
        ac_space=env.action_space,
        hid_size=args.hidden_size,
        num_hid_layers=args.num_layers
    )
    
    # Create SFT trainer
    trainer = MotionImitationSFT(
        env=env,
        policy=policy,
        mocap_path=args.mocap,
        device=args.device
    )
    
    # Extract dataset
    dataset = trainer.extract_dataset(
        lookahead_frames=args.lookahead,
        refine_actions=args.refine_actions,
        refine_method=args.refine_method,
        refine_iterations=args.refine_iterations
    )
    
    # Train
    trainer.train(
        dataset=dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_split=args.val_split,
        save_path=args.save_path
    )
    
    # Evaluate
    if not args.no_eval:
        trainer.evaluate_in_environment(
            num_episodes=args.eval_episodes,
            max_steps=args.eval_steps
        )
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"1. Test the policy: python dp_env_v3.py --load_model {args.save_path}")
    print(f"2. Fine-tune with RL: python trpo_torch.py --task train --load_sft_pretrain {args.save_path}")
    print()


if __name__ == '__main__':
    main()
