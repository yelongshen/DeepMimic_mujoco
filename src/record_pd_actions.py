#!/usr/bin/env python3
"""
Record video of PD control actions reproducing mocap motion

This visualizes how well the extracted actions match the mocap trajectory.
Useful for debugging and verifying action quality before training.
"""

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from dp_env_v3 import DPEnv
from deepmimic_mujoco.mocap_v2 import MocapDM


def compute_pd_action(qpos_current, qvel_current, qpos_target, kp=1.0, kd=0.1):
    """Compute PD control action"""
    current_joints = qpos_current[7:]
    target_joints = qpos_target[7:]
    joint_vels = qvel_current[6:]
    
    action = kp * (target_joints - current_joints) - kd * joint_vels
    action = np.clip(action, -1.0, 1.0)
    
    return action


def record_pd_actions_video(mocap_path, output_path='render/pd_actions_video.avi',
                            duration=60.0, kp=1.0, kd=0.1, 
                            show_comparison=False, refined=False,
                            refine_iterations=3):
    """
    Record video of agent following mocap using PD control actions
    
    Args:
        mocap_path: Path to mocap file
        output_path: Where to save video
        duration: Video duration in seconds
        kp, kd: PD control gains
        show_comparison: If True, show mocap vs simulated side-by-side
        refined: Whether to use refined actions
        refine_iterations: Number of refinement iterations
    """
    print("="*70)
    print("Recording Video of PD Control Actions")
    print("="*70)
    print(f"Mocap file: {mocap_path}")
    print(f"Output: {output_path}")
    print(f"Duration: {duration}s")
    print(f"PD gains: Kp={kp}, Kd={kd}")
    if refined:
        print(f"Using refined actions ({refine_iterations} iterations)")
    print()
    
    # Create environment
    env = DPEnv()
    
    # Load mocap
    mocap = MocapDM()
    mocap.load_mocap(mocap_path)
    
    print(f"Loaded mocap: {len(mocap.data_config)} frames")
    print(f"Mocap dt: {mocap.dt}s")
    print()
    
    # Compute actions
    print("Computing PD control actions...")
    actions = []
    for i in tqdm(range(len(mocap.data_config) - 1)):
        qpos_current = mocap.data_config[i]
        qvel_current = mocap.data_vel[i]
        qpos_target = mocap.data_config[i + 1]
        
        action = compute_pd_action(qpos_current, qvel_current, qpos_target, kp, kd)
        actions.append(action)
    
    actions = np.array(actions)
    print(f"Computed {len(actions)} actions")
    
    # Optional: Refine actions
    if refined:
        print(f"\nRefining actions ({refine_iterations} iterations)...")
        actions = refine_actions_feedback(env, mocap, actions, refine_iterations)
    
    # Action statistics
    print(f"\nAction statistics:")
    print(f"  Mean: {actions.mean():.4f}")
    print(f"  Std:  {actions.std():.4f}")
    print(f"  Min:  {actions.min():.4f}")
    print(f"  Max:  {actions.max():.4f}")
    clipped = np.sum((actions == -1.0) | (actions == 1.0))
    print(f"  Clipped: {clipped}/{actions.size} ({100*clipped/actions.size:.1f}%)")
    
    # Setup recording
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # Enable rendering
    env.enable_rendering()
    
    # Initialize video writer
    from cv2 import VideoWriter, VideoWriter_fourcc
    fps = int(1.0 / mocap.dt)
    fourcc = VideoWriter_fourcc(*'XVID')
    video_writer = VideoWriter(output_path, fourcc, fps, (env.width, env.height))
    
    print(f"\n{'='*70}")
    print(f"Recording video...")
    print(f"{'='*70}")
    print(f"FPS: {fps}")
    print(f"Resolution: {env.width}x{env.height}")
    print()
    
    # Set initial state
    env.set_state(mocap.data_config[0], mocap.data_vel[0])
    
    # Calculate number of steps
    num_steps = int(duration / mocap.dt)
    num_mocap_frames = len(mocap.data_config)
    
    # Track statistics
    total_reward = 0
    rewards = []
    joint_errors = []
    root_errors = []
    
    print(f"Simulating and recording {num_steps} steps...")
    
    for step in tqdm(range(num_steps)):
        # Get action (loop mocap if needed)
        action_idx = step % len(actions)
        action = actions[action_idx]
        
        # Apply action
        obs, reward, done, info = env.step(action)
        total_reward += reward
        rewards.append(reward)
        
        # Compute tracking error
        mocap_idx = (step + 1) % num_mocap_frames
        qpos_actual = env.sim.data.qpos.copy()
        qpos_target = mocap.data_config[mocap_idx]
        
        joint_error = np.abs(qpos_actual[7:] - qpos_target[7:]).mean()
        root_error = np.linalg.norm(qpos_actual[:3] - qpos_target[:3])
        
        joint_errors.append(joint_error)
        root_errors.append(root_error)
        
        # Render frame
        frame = env.render()
        
        # Optionally add text overlay with stats
        if frame is not None:
            import cv2
            # Add info text
            info_text = [
                f"Step: {step}/{num_steps}",
                f"Reward: {reward:.2f}",
                f"Joint Error: {joint_error:.4f} rad",
                f"Root Error: {root_error:.3f} m",
            ]
            
            y_offset = 30
            for i, text in enumerate(info_text):
                cv2.putText(frame, text, (10, y_offset + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            video_writer.write(frame)
        
        # Reset if done (shouldn't happen for mocap tracking)
        if done:
            print(f"\n⚠️  Episode ended at step {step}")
            break
    
    # Cleanup
    video_writer.release()
    
    # Print summary
    print(f"\n{'='*70}")
    print("Recording Complete!")
    print(f"{'='*70}")
    print(f"\nVideo saved to: {output_path}")
    print(f"\nPerformance Statistics:")
    print(f"  Total steps:       {len(rewards)}")
    print(f"  Mean reward:       {np.mean(rewards):.4f}")
    print(f"  Total reward:      {total_reward:.2f}")
    print(f"  Mean joint error:  {np.mean(joint_errors):.6f} rad")
    print(f"  Max joint error:   {np.max(joint_errors):.6f} rad")
    print(f"  Mean root error:   {np.mean(root_errors):.6f} m")
    print(f"  Max root error:    {np.max(root_errors):.6f} m")
    
    # Assessment
    mean_joint_err = np.mean(joint_errors)
    mean_root_err = np.mean(root_errors)
    
    print(f"\nAssessment:")
    if mean_joint_err < 0.1 and mean_root_err < 0.1:
        print("  ✓ EXCELLENT: Actions reproduce mocap very well!")
    elif mean_joint_err < 0.2 and mean_root_err < 0.3:
        print("  ✓ GOOD: Actions track mocap reasonably well")
    elif mean_joint_err < 0.5:
        print("  ~ OK: Some drift, consider refinement or tuning Kp/Kd")
    else:
        print("  ⚠️  WARNING: Large tracking error, adjust PD gains")
    
    print(f"\n{'='*70}")
    print(f"View video: {output_path}")
    print(f"{'='*70}\n")


def refine_actions_feedback(env, mocap, actions, iterations=3, alpha=0.5):
    """Refine actions using error feedback"""
    print(f"Refining with feedback correction (alpha={alpha})...")
    
    for iter_num in range(iterations):
        # Reset environment
        env.set_state(mocap.data_config[0], mocap.data_vel[0])
        
        errors = []
        
        for i in range(len(actions)):
            # Apply action
            env.step(actions[i])
            
            # Measure error
            qpos_actual = env.sim.data.qpos
            qpos_target = mocap.data_config[i + 1]
            joint_error = qpos_target[7:] - qpos_actual[7:]
            
            errors.append(np.abs(joint_error).mean())
            
            # Correct action
            correction = alpha * joint_error
            actions[i] = np.clip(actions[i] + correction, -1.0, 1.0)
        
        print(f"  Iteration {iter_num+1}/{iterations}: mean error = {np.mean(errors):.6f} rad")
    
    return actions


def main():
    parser = argparse.ArgumentParser(description='Record video of PD control actions')
    
    parser.add_argument('--mocap', type=str,
                       default='deepmimic_mujoco/motions/humanoid3d_dance_a.txt',
                       help='Path to mocap file')
    parser.add_argument('--output', type=str,
                       default='render/pd_actions_video.avi',
                       help='Output video path')
    parser.add_argument('--duration', type=float, default=60.0,
                       help='Video duration in seconds')
    parser.add_argument('--kp', type=float, default=1.0,
                       help='Proportional gain')
    parser.add_argument('--kd', type=float, default=0.1,
                       help='Derivative gain')
    parser.add_argument('--refined', action='store_true',
                       help='Use refined actions')
    parser.add_argument('--refine_iterations', type=int, default=3,
                       help='Number of refinement iterations')
    
    args = parser.parse_args()
    
    record_pd_actions_video(
        mocap_path=args.mocap,
        output_path=args.output,
        duration=args.duration,
        kp=args.kp,
        kd=args.kd,
        refined=args.refined,
        refine_iterations=args.refine_iterations
    )


if __name__ == '__main__':
    main()
