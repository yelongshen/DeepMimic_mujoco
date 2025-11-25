#!/usr/bin/env python3
import os
import sys

# Add current directory to Python path for local imports
# This ensures we can import from the local 'mujoco' directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Set MuJoCo rendering backend for headless/xvfb environments
# This must be done before any MuJoCo imports
display = os.environ.get('DISPLAY', '')
mujoco_gl = os.environ.get('MUJOCO_GL', '')

# If MUJOCO_GL is already set, respect it
if not mujoco_gl:
    # If DISPLAY is :99 (typical Xvfb), use GLFW
    # If DISPLAY is empty or not set, use EGL for true headless
    if display.startswith(':99'):
        os.environ['MUJOCO_GL'] = 'glfw'
        print("Detected Xvfb display - using GLFW rendering")
    elif not display:
        os.environ['MUJOCO_GL'] = 'egl'
        print("Warning: No DISPLAY available. Using EGL for headless rendering.")
    else:
        # Normal display available
        os.environ['MUJOCO_GL'] = 'glfw'
        print(f"Display detected: {display} - using GLFW rendering")
else:
    print(f"Using pre-configured MUJOCO_GL={mujoco_gl}")

import numpy as np
import math
import random
from os import getcwd

# Import from local deepmimic_mujoco directory (renamed from 'mujoco' to avoid conflict)
from deepmimic_mujoco.mocap_v2 import MocapDM
from deepmimic_mujoco.mujoco_interface import MujocoInterface
from deepmimic_mujoco.mocap_util import JOINT_WEIGHT
# Updated to use modern mujoco package with compatibility wrapper
from mujoco_py_compat import load_model_from_xml, MjSim, MjViewer

try:
    import gymnasium as gym
    from gymnasium import spaces
    from gymnasium import utils
except ImportError:
    import gym
    from gym import spaces
    from gym import utils

from config import Config
from pyquaternion import Quaternion

from transformations import quaternion_from_euler

BODY_JOINTS = ["chest", "neck", "right_shoulder", "right_elbow", 
            "left_shoulder", "left_elbow", "right_hip", "right_knee", 
            "right_ankle", "left_hip", "left_knee", "left_ankle"]

DOF_DEF = {"root": 3, "chest": 3, "neck": 3, "right_shoulder": 3, 
           "right_elbow": 1, "right_wrist": 0, "left_shoulder": 3, "left_elbow": 1, 
           "left_wrist": 0, "right_hip": 3, "right_knee": 1, "right_ankle": 3, 
           "left_hip": 3, "left_knee": 1, "left_ankle": 3}

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

class DPEnv(gym.Env, utils.EzPickle):
    def __init__(self):
        xml_file_path = Config.xml_path

        self.mocap = MocapDM()
        self.interface = MujocoInterface()
        print('mocap_path', Config.mocap_path)
        self.load_mocap(Config.mocap_path)

        self.weight_pose = 0.5
        self.weight_vel = 0.05
        self.weight_root = 0.2
        self.weight_end_eff = 0.15
        self.weight_com = 0.1

        self.scale_pose = 2.0
        self.scale_vel = 0.1
        self.scale_end_eff = 40.0
        self.scale_root = 5.0
        self.scale_com = 10.0
        self.scale_err = 1.0

        self.reference_state_init()
        self.idx_curr = -1
        self.idx_tmp_count = -1

        # Load MuJoCo model
        print('xml_file_path', xml_file_path)
        with open(xml_file_path, 'r') as f:
            xml_string = f.read()
        
        #print('xml_string', xml_string)
        
        model = load_model_from_xml(xml_string)
        self.sim = MjSim(model)
        self.model = self.sim.model
        self.data = self.sim.data
        self.viewer = None
        self.frame_skip = 6
        
        # Initialize state
        self.init_qpos = self.sim.data.qpos.copy()
        self.init_qvel = self.sim.data.qvel.copy()
        
        # Define action and observation spaces
        obs = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float64
        )
        
        # Action space (26 actuators for humanoid)
        bounds = self.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # Random number generator
        self.np_random = np.random.RandomState()
        
        # Initialize step length (used in step() and reset_model())
        self.step_len = 1
        
        utils.EzPickle.__init__(self)

    def do_simulation(self, ctrl, n_frames):
        """Run simulation for n_frames with the given control."""
        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()
    
    def set_state(self, qpos, qvel):
        """Set the state of the simulation."""
        old_state = self.sim.get_state()
        new_state = np.concatenate([qpos, qvel])
        self.sim.set_state(new_state)
        self.sim.forward()
    
    def reset(self):
        """Reset the environment to initial state."""
        self.sim.reset()
        return self.reset_model()
    
    def seed(self, seed=None):
        """Set the seed for this environment's random number generator."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            self.np_random = np.random.RandomState(seed)
        return [seed]
    
    def render(self, mode='human'):
        """Render the environment."""
        if mode == 'human':
            # Check if we have a display
            import os
            if 'DISPLAY' not in os.environ or not os.environ['DISPLAY']:
                print("Warning: No DISPLAY available. Rendering in offscreen mode.")
                print("To see the visualization, either:")
                print("  1. Run on a system with a display")
                print("  2. Use X11 forwarding: ssh -X user@host")
                print("  3. Use xvfb: xvfb-run python dp_env_v3.py")
                # Fall back to offscreen rendering
                return self.sim.render(width=640, height=480, mode='offscreen')
            else:
                if self.viewer is None:
                    self.viewer = MjViewer(self.sim)
                self.viewer.render()
                # Also return the frame for video capture
                return self.sim.render(width=640, height=480, mode='offscreen')
        
        if mode == 'rgb_array':
            return self.sim.render(width=640, height=480, mode='offscreen')

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()[7:] # ignore root joint
        velocity = self.sim.data.qvel.flat.copy()[6:] # ignore root joint
        return np.concatenate((position, velocity))

    def reference_state_init(self):
        self.idx_init = random.randint(0, self.mocap_data_len-1)
        # self.idx_init = 0
        self.idx_curr = self.idx_init
        self.idx_tmp_count = 0

    def early_termination(self):
        pass

    def get_joint_configs(self):
        data = self.sim.data
        return data.qpos[7:] # to exclude root joint

    def load_mocap(self, filepath):
        self.mocap.load_mocap(filepath)
        self.mocap_dt = self.mocap.dt
        self.mocap_data_len = len(self.mocap.data)

        print('mocap_data_len', self.mocap_data_len)
        print('mocap_dt', self.mocap_dt)
        
    def calc_config_errs(self, env_config, mocap_config):
        assert len(env_config) == len(mocap_config)
        return np.sum(np.abs(env_config - mocap_config))

    def calc_config_reward(self):
        assert len(self.mocap.data) != 0
        err_configs = 0.0

        target_config = self.mocap.data_config[self.idx_curr][7:] # to exclude root joint
        self.curr_frame = target_config
        curr_config = self.get_joint_configs()

        err_configs = self.calc_config_errs(curr_config, target_config)
        
        # FIXED: Normalize error by number of joints and scale reward for better learning
        num_joints = len(target_config)
        normalized_err = err_configs / num_joints  # Average error per joint
        
        # Use scaled exponential: exp(-k * error) where k controls sensitivity
        # k=2.0 means reward drops to ~0.14 when avg error is 1 radian per joint
        reward_config = math.exp(-2.0 * normalized_err)
        
        # Scale to 0-10 range to provide stronger learning signal
        reward_config = reward_config * 10.0

        self.idx_curr += 1
        self.idx_curr = self.idx_curr % self.mocap_data_len

        return reward_config

    def step(self, action):
        # self.step_len = int(self.mocap_dt // self.model.opt.timestep)
        self.step_len = 1
        # step_times = int(self.mocap_dt // self.model.opt.timestep)
        step_times = 1
        # pos_before = mass_center(self.model, self.sim)
        self.do_simulation(action, step_times)
        # pos_after = mass_center(self.model, self.sim)

        observation = self._get_obs()

        reward_alive = 1.0
        # Enable motion imitation reward (compares agent motion to reference mocap)
        reward = self.calc_config_reward()
        
        # Optional: use full reward with action penalty and forward movement
        # Uncomment below for more complete reward:
        '''
        reward_obs = self.calc_config_reward()
        reward_acs = -0.1 * np.square(self.sim.data.ctrl).sum()
        reward_forward = 0.25*(pos_after - pos_before)
        reward = reward_obs + reward_acs + reward_forward + reward_alive
        info = dict(reward_obs=reward_obs, reward_acs=reward_acs, reward_forward=reward_forward)
        '''
        
        info = dict()
        done = self.is_done()

        return observation, reward, done, info

    def is_done(self):
        mass = np.expand_dims(self.model.body_mass, 1)
        xpos = self.sim.data.xipos
        z_com = (np.sum(mass * xpos, 0) / np.sum(mass))[2]
        done = bool((z_com < 0.7) or (z_com > 2.0))
        return done

    def goto(self, pos):
        self.sim.data.qpos[:] = pos[:]
        self.sim.forward()

    def get_time(self):
        return self.sim.data.time

    def reset_model(self):
        self.reference_state_init()
        qpos = self.mocap.data_config[self.idx_init]
        qvel = self.mocap.data_vel[self.idx_init]
        # qvel = self.init_qvel
        self.set_state(qpos, qvel)
        observation = self._get_obs()
        self.idx_tmp_count = -self.step_len
        return observation

    def reset_model_init(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self._get_obs()

    def viewer_setup(self):
        pass
        # self.viewer.cam.trackbodyid = 1
        # self.viewer.cam.distance = self.model.stat.extent * 1.0
        # self.viewer.cam.lookat[2] = 2.0
        # self.viewer.cam.elevation = -20

if __name__ == "__main__":
    import os
    import time
    import argparse
    import torch
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='DeepMimic environment with video recording')
    parser.add_argument('--load_model', type=str, default=None,
                       help='Path to trained policy model (e.g., policy_sft_pretrained.pth)')
    parser.add_argument('--time_limit', type=float, default=60.0,
                       help='Time limit in seconds (default: 60)')
    parser.add_argument('--stochastic', action='store_true',
                       help='Use stochastic policy (default: deterministic)')
    args = parser.parse_args()
    
    # initialize environment
    env = DPEnv()
    env.reset_model()
    
    # Load policy if provided
    policy = None
    if args.load_model:
        print(f"\nLoading trained policy from: {args.load_model}")
        try:
            from mlp_policy_torch import MlpPolicy
            policy = MlpPolicy(env.observation_space, env.action_space)
            policy.load_state_dict(torch.load(args.load_model, map_location='cpu'))
            policy.eval()
            print("✓ Policy loaded successfully!")
            print(f"  Stochastic: {args.stochastic}")
        except Exception as e:
            print(f"✗ Failed to load policy: {e}")
            print("  Continuing with mocap replay (no policy)")
            policy = None
    else:
        print("\nNo policy loaded - will replay mocap motion")



    import cv2
    from VideoSaver import VideoSaver
    width = 640
    height = 480
    fps = 30  # Frames per second for video

    # Initialize video saver - video_path=None will auto-generate path in ./render/
    vid_save = VideoSaver(width=width, height=height, fps=fps, video_path=None)
    print(f"Video recording enabled. Saving to ./render/ directory")

    # env.load_mocap("/home/mingfei/Documents/DeepMimic/mujoco/motions/humanoid3d_crawl.txt")
    action_size = env.action_space.shape[0]
    ac = np.zeros(action_size)
    
    print('action space', action_size)

    # Check if running in headless mode
    headless = 'DISPLAY' not in os.environ or not os.environ['DISPLAY']
    if headless:
        print("Running in headless mode (no display detected)")
        print("Using offscreen rendering for video capture")
    else:
        print("Display detected - will show visualization window")
    
    # Time limit: 1 minute (60 seconds)
    time_limit = 60.0
    start_time = time.time()
    frame_count = 0
    
    # Print dimensions on first iteration
    print(f"Starting simulation with {args.time_limit} second time limit...")
    print(f"\nMocap data dimensions:")
    print(f"  Total frames: {len(env.mocap.data_config)}")
    qpos_sample = env.mocap.data_config[0]
    qvel_sample = env.mocap.data_vel[0]
    print(f"  qpos dimension: {len(qpos_sample)} - {qpos_sample.shape}")
    print(f"  qvel dimension: {len(qvel_sample)} - {qvel_sample.shape}")
    print(f"\nMuJoCo model dimensions:")
    print(f"  model.nq (position DOF): {env.model.nq}")
    print(f"  model.nv (velocity DOF): {env.model.nv}")
    print(f"  model.nu (actuators): {env.model.nu}")
    print(f"  sim.data.qpos shape: {env.sim.data.qpos.shape}")
    print(f"  sim.data.qvel shape: {env.sim.data.qvel.shape}")
    
    print(f"\nAction space:")
    print(f"  action_space shape: {env.action_space.shape}")
    print(f"  action dimension: {env.action_space.shape[0]}")
    print(f"  action bounds: low={env.action_space.low[:5]}... high={env.action_space.high[:5]}...")
    
    print(f"\nObservation space:")
    print(f"  observation_space shape: {env.observation_space.shape}")
    print(f"  observation dimension: {env.observation_space.shape[0]}")
    
    print(f"\n{'='*60}")
    print("MODE: {'POLICY CONTROL' if policy else 'MOCAP REPLAY'}")
    print(f"{'='*60}\n")
    
    # Reset environment to start
    obs = env.reset()
    total_reward = 0
    step_count = 0
    
    while True:
        # Check time limit
        elapsed_time = time.time() - start_time
        if elapsed_time >= time_limit:
            print(f"\nTime limit reached ({time_limit} seconds)")
            print(f"Total frames captured: {frame_count}")
            if policy:
                print(f"Total steps: {step_count}")
                print(f"Total reward: {total_reward:.2f}")
                print(f"Average reward: {total_reward/step_count:.4f}")
            break
        
        if policy:
            # Use policy to generate actions
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                action, _ = policy.act(obs_tensor, stochastic=args.stochastic)
            
            # Step environment with policy action
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            if done:
                print(f"  Episode ended at step {step_count}, reward: {total_reward:.2f}")
                obs = env.reset()
                total_reward = 0
                step_count = 0
        else:
            # Replay mocap motion (original behavior)
            qpos = env.mocap.data_config[env.idx_curr]
            qvel = env.mocap.data_vel[env.idx_curr]
            env.set_state(qpos, qvel)
            env.sim.step()
            env.calc_config_reward()
        
        # Render and capture frame for video (works in both headless and display modes)
        # render() now always returns a frame, and shows window if display is available
        frame = env.render(mode='human')
        
        # Resize frame to match video dimensions and save
        if frame is not None:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # Resize if needed
            if frame_bgr.shape[0] != height or frame_bgr.shape[1] != width:
                frame_bgr = cv2.resize(frame_bgr, (width, height))
            vid_save.addFrame(frame_bgr)
            frame_count += 1
            
            # Print progress every 30 frames (~1 second at 30fps)
            if frame_count % 30 == 0:
                print(f"Progress: {elapsed_time:.1f}s / {time_limit}s - Frames: {frame_count}")

    vid_save.close()
    print("Video saved successfully!")
