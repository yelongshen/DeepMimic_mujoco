# Quick Start: Video Recording

## Run with Video Recording (1 minute)

### On Headless Server (WSL2/Linux without display):

```bash
# Make script executable
chmod +x run_video.sh

# Run (will create video in src/render/)
./run_video.sh
```

Or manually:
```bash
cd src
xvfb-run -a -s "-screen 0 1400x900x24" python dp_env_v3.py
```

### On System with Display:

```bash
cd src
python dp_env_v3.py
```

## What Happens:

1. ✅ Simulation runs for **60 seconds** (1 minute)
2. ✅ Video is saved to `src/render/` with timestamp (e.g., `20251118_123708.avi`)
3. ✅ Progress updates printed every second
4. ✅ Works on both headless and display systems

## Output:

```
Detected Xvfb display - using GLFW rendering
Video recording enabled. Saving to ./render/ directory
Running in headless mode (no display detected)
Using offscreen rendering for video capture
Starting simulation with 60.0 second time limit...
Progress: 1.0s / 60.0s - Frames: 30
Progress: 2.0s / 60.0s - Frames: 60
...
Time limit reached (60.0 seconds)
Total frames captured: 1800
Video saved successfully!
```

## Configuration:

Edit `src/dp_env_v3.py` to change:

```python
# Line ~271: Change duration
time_limit = 120.0  # 2 minutes instead of 1

# Line ~266: Change video settings
width = 1280
height = 720
fps = 60  # Higher frame rate
```

## Troubleshooting:

### "xvfb-run: command not found"
```bash
sudo apt-get install -y xvfb
```

### "No such file or directory: './render/'"
The script automatically creates the directory. Check permissions:
```bash
mkdir -p src/render
chmod 755 src/render
```

### Video file is very small or empty
Check the terminal output for errors during rendering.

### Want different video format?
Edit `src/VideoSaver.py` line 27:
```python
# Change from MJPG to MP4V
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# And change filename extension from .avi to .mp4
```

## Advanced: Custom Mocap Motion

To record a different motion:

```python
# Edit src/dp_env_v3.py, uncomment line ~275:
env.load_mocap("/path/to/your/motion.txt")
```
