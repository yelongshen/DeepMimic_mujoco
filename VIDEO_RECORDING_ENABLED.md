# Video Recording Now Enabled! 🎥

## What Changed

Video recording has been integrated into the TRPO evaluation pipeline. Videos are now automatically saved when you run policy evaluation.

## Changes Made

### 1. **trpo.py - `runner()` function**
- Added VideoSaver initialization with timestamp
- Creates `./render/` directory automatically
- Passes video_saver to trajectory generator
- Saves video after all trajectories complete

### 2. **trpo.py - `traj_1_generator()` function**
- Added `video_saver` parameter (optional, defaults to None)
- Captures frames using `env.render(mode='rgb_array')`
- Adds each frame to video during rollout

### 3. **VideoSaver.py - Enhanced**
- Added `add_frame()` method (alias for `addFrame()`)
- Added `save()` method for explicit saving
- Automatic RGB to BGR conversion for OpenCV
- Frame resizing if needed
- Better error handling and logging
- Frame counter for statistics

## How to Use

### Run Evaluation with Video Recording

```bash
cd /mnt/c/DeepMimic_mujoco/src
./run_eval_hide_cuda.sh
```

Or manually:

```bash
cd /mnt/c/DeepMimic_mujoco/src
export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL="glfw"
xvfb-run -a python trpo.py --task evaluate \
  --load_model_path checkpoint_tmp/DeepMimic/trpo-walk-0/DeepMimic/trpo-walk-0 \
  --stochastic_policy
```

### Output

Videos are saved to:
```
src/render/
  ├── eval_20251120_134520.avi  # Timestamp format: YYYYMMDD_HHMMSS
  ├── eval_20251120_135612.avi
  └── ...
```

### Video Details

- **Format**: AVI (MJPEG codec)
- **FPS**: 30 frames per second
- **Resolution**: 640x480 pixels
- **Duration**: Depends on trajectory length (100 trajectories by default)

## Benefits

✅ **Automatic Recording**: No manual intervention needed
✅ **Timestamped Files**: Each run creates unique file
✅ **Full Trajectories**: Records all evaluation episodes
✅ **Headless Compatible**: Works with xvfb virtual display
✅ **No NumPy Files**: Clean video output instead of data dumps

## Troubleshooting

### No frames captured
- Make sure `env.render(mode='rgb_array')` returns valid frames
- Check that MUJOCO_GL is set correctly ("glfw" for xvfb, "egl" for hardware)

### Video file is empty
- Verify xvfb is running: `xvfb-run -a ...`
- Check render directory permissions: `ls -la ./render/`

### OpenCV errors
- Install OpenCV: `pip install opencv-python`
- Try different codec if MJPG fails

### Display warnings
- Expected in headless mode - video still saves correctly
- Use `xvfb-run` to suppress DISPLAY warnings

## Example Output

```
Recording video to: ./render/eval_20251120_134520.avi
VideoSaver: Opening ./render/eval_20251120_134520.avi
VideoSaver: Initialized successfully
100%|████████████████████| 100/100 [02:30<00:00,  1.50s/it]
VideoSaver: Saved 15000 frames to ./render/eval_20251120_134520.avi
Video saved to: ./render/eval_20251120_134520.avi
Average length: 150.0
Average return: 5.2
```

## Playing Videos

### On Linux/WSL:
```bash
# Install VLC or mpv
sudo apt-get install vlc
vlc ./render/eval_20251120_134520.avi

# Or use mpv
sudo apt-get install mpv
mpv ./render/eval_20251120_134520.avi
```

### On Windows:
```powershell
# Open in default video player
start .\src\render\eval_20251120_134520.avi

# Or use VLC
"C:\Program Files\VideoLAN\VLC\vlc.exe" .\src\render\eval_20251120_134520.avi
```

### Copy to Windows from WSL:
```bash
# From WSL terminal
cp ./render/eval_20251120_134520.avi /mnt/c/Users/YourUsername/Videos/
```

## Next Steps

- ✅ Videos save automatically during evaluation
- 🎯 Run evaluation to generate videos
- 📊 Analyze trained policy performance visually
- 🔄 Compare different checkpoints by watching their videos

Enjoy your motion imitation videos! 🎬
