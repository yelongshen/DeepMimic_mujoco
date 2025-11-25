# Recording Videos of PD Control Actions

## Quick Start

### Record Standard PD Actions

```bash
cd src
bash record_pd_actions.sh
```

This creates: `render/pd_actions_video.avi` (60 seconds)

---

### Compare Standard vs Refined Actions

```bash
cd src
bash compare_pd_refinement.sh
```

This creates:
- `render/pd_standard.avi` (standard PD control)
- `render/pd_refined.avi` (after 3 refinement iterations)

Compare both to see the improvement!

---

## Custom Recording

### Python Script Options

```bash
cd src

# Basic recording
python record_pd_actions.py

# Custom duration
python record_pd_actions.py --duration 30.0

# Different PD gains
python record_pd_actions.py --kp 1.5 --kd 0.2

# With refinement
python record_pd_actions.py --refined --refine_iterations 5

# Different mocap file
python record_pd_actions.py --mocap deepmimic_mujoco/motions/humanoid3d_walk.txt

# Custom output location
python record_pd_actions.py --output my_videos/dance_pd.avi
```

---

## What the Video Shows

### On-Screen Display

Each frame shows:
```
Step: 123/1000
Reward: 8.45
Joint Error: 0.045 rad
Root Error: 0.023 m
```

### Color-Coded Assessment

At the end, you'll see:
- ✅ **GREEN (Excellent)**: Joint error < 0.1 rad, Root error < 0.1 m
- ✅ **BLUE (Good)**: Joint error < 0.2 rad, Root error < 0.3 m
- ⚠️ **YELLOW (OK)**: Some drift visible
- ❌ **RED (Poor)**: Large tracking errors

---

## Interpreting Results

### Good PD Actions (Ready for SFT)
```
Performance Statistics:
  Mean reward:       8.2341
  Mean joint error:  0.045 rad
  Mean root error:   0.023 m
  Max root error:    0.150 m

Assessment:
  ✓ EXCELLENT: Actions reproduce mocap very well!
```

**What you'll see in video:**
- Smooth motion matching mocap
- Character stays in place (minimal drift)
- Natural-looking movements

---

### Poor PD Actions (Need Tuning)
```
Performance Statistics:
  Mean reward:       5.1234
  Mean joint error:  0.345 rad
  Mean root error:   0.678 m
  Max root error:    1.500 m

Assessment:
  ⚠️ WARNING: Large tracking error, adjust PD gains
```

**What you'll see in video:**
- Jerky or unnatural movements
- Character drifts away from starting position
- Poses don't match mocap well

**Solutions:**
1. Reduce Kp: `python record_pd_actions.py --kp 0.5`
2. Increase Kd: `python record_pd_actions.py --kd 0.2`
3. Use refinement: `python record_pd_actions.py --refined`

---

## Comparison: Standard vs Refined

### Expected Differences

**Standard PD (Kp=1.0, Kd=0.1):**
```
Mean joint error:  0.29 rad
Max root error:    1.1 m
```
- Some drift after 10-20 seconds
- Character may move 1 meter from start

**Refined (3 iterations):**
```
Mean joint error:  0.15 rad  (48% improvement!)
Max root error:    0.5 m     (55% improvement!)
```
- Much less drift
- Character stays closer to origin
- Smoother motion

---

## Use Cases

### 1. Verify PD Gains Are Good

Before training SFT:
```bash
python record_pd_actions.py --duration 30
```

Check video - if tracking looks good, proceed with training!

### 2. Compare Different PD Gains

```bash
# Test conservative gains
python record_pd_actions.py --kp 0.5 --kd 0.2 --output render/pd_conservative.avi

# Test aggressive gains
python record_pd_actions.py --kp 1.5 --kd 0.05 --output render/pd_aggressive.avi

# Compare videos to find best gains
```

### 3. Demonstrate Refinement Impact

```bash
bash compare_pd_refinement.sh
```

Watch both videos to see how refinement reduces drift.

### 4. Debug Mocap Issues

```bash
# Record long video to see cumulative errors
python record_pd_actions.py --duration 120
```

If video shows large drift, may need refinement or different gains.

---

## Troubleshooting

### Video Not Created

**Problem:** No video file appears

**Solutions:**
1. Check render directory exists:
   ```bash
   mkdir -p render
   ```

2. Check OpenCV is installed:
   ```bash
   pip install opencv-python
   ```

3. Try running in virtual display:
   ```bash
   xvfb-run -a python record_pd_actions.py
   ```

### Video Is Black/Blank

**Problem:** Video file exists but shows nothing

**Solutions:**
1. Rendering may not be enabled - the script handles this automatically
2. Try different codec:
   - Edit `record_pd_actions.py` line with `VideoWriter_fourcc`
   - Change `'XVID'` to `'MJPG'` or `'mp4v'`

### "No display" Error

**Problem:** Error about DISPLAY not set

**Solution:** Use xvfb:
```bash
xvfb-run -a -s "-screen 0 1400x900x24" python record_pd_actions.py
```

Or use the provided bash scripts (already includes xvfb).

---

## Video Specifications

```
Format:     AVI (XVID codec)
Resolution: 1400x900 pixels
FPS:        30 (matches mocap timestep)
Duration:   Configurable (default 60s)
```

---

## Advanced: Frame-by-Frame Analysis

To extract individual frames:

```bash
# Use ffmpeg to extract frames
ffmpeg -i render/pd_actions_video.avi -vf fps=1 frames/frame_%04d.png
```

Then analyze specific frames where errors are high.

---

## Workflow Integration

### Recommended Process:

1. **Record PD actions video:**
   ```bash
   python record_pd_actions.py --duration 30
   ```

2. **Check video quality:**
   - Does motion look natural?
   - How much drift occurs?
   - Are errors acceptable?

3. **If needed, tune or refine:**
   ```bash
   # Option A: Tune gains
   python record_pd_actions.py --kp 0.7 --kd 0.15
   
   # Option B: Use refinement
   python record_pd_actions.py --refined
   ```

4. **Once satisfied, train SFT:**
   ```bash
   python train_sft.py --epochs 100
   ```

5. **Compare SFT policy video** (after training):
   ```bash
   python dp_env_v3.py --load_model policy_sft_pretrained.pth
   ```

---

## Summary

**Quick commands:**

```bash
# Record standard PD actions
bash record_pd_actions.sh

# Compare standard vs refined
bash compare_pd_refinement.sh

# Custom recording with options
python record_pd_actions.py --duration 30 --refined --output my_video.avi
```

**What to look for:**
- ✅ Smooth, natural motion
- ✅ Minimal drift (< 0.5m)
- ✅ Good pose matching
- ✅ High rewards (> 7.0)

**Use videos to:**
- Verify PD gains before training
- Compare refinement methods
- Debug tracking issues
- Demonstrate results

Enjoy your videos! 🎥
