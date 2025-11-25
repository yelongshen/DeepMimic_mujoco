import numpy as np
import os
import time
import cv2
from dp_env_v3 import DPEnv
from VideoSaver import VideoSaver

if __name__ == "__main__":
    # Initialize environment
    env = DPEnv()
    env.reset_model()

    action_size = env.action_space.shape[0]
    ac = np.ones(action_size)
    
    np.set_printoptions(precision=3)

    # Video recording setup
    width = 640
    height = 480
    fps = 30

    # Initialize video saver
    vid_save = VideoSaver(width=width, height=height, fps=fps, dumpDir="./render")
    print(f"Video recording enabled. Saving to ./render/ directory")

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

    # Print dimensions
    print(f"\nStarting torque control test with {time_limit} second time limit...")
    print(f"Action space: {env.action_space.shape[0]} dimensions")
    print(f"Control mode: PD controller tracking mocap reference\n")

    while True:
        # Check time limit
        elapsed_time = time.time() - start_time
        if elapsed_time >= time_limit:
            print(f"\nTime limit reached ({time_limit} seconds)")
            print(f"Total frames captured: {frame_count}")
            break

        # PD controller to track mocap reference
        target_config = env.mocap.data_config[env.idx_curr][7:] # to exclude root joint
        target_config_vel = env.mocap.data_vel[env.idx_curr][6:]
        curr_config = env.sim.data.qpos[7:]
        curr_config_vel = env.sim.data.qvel[6:]
        
        # PD control: proportional term
        ac = 0.8 * np.array(target_config - curr_config)
        # Optional: add derivative term
        # ac += 0.02 * np.array(target_config_vel - curr_config_vel)

        # Step environment with computed torques
        _, rew, done, info = env.step(ac)
        if done:
            env.reset_model()

        # Render and capture frame for video
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
                print(f"Progress: {elapsed_time:.1f}s / {time_limit}s - Frames: {frame_count} - Reward: {rew:.3f}")

    vid_save.close()
    print("Video saved successfully!")