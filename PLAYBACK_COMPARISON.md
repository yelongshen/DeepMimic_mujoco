# Comparison: Kinematic Playback vs Torque Control

This project has two different ways to replay mocap data:

## 1. Kinematic Playback (`dp_env_v3.py`)

**What it does:**
- Directly sets joint positions from mocap data
- No physics simulation involved
- Perfect reproduction of original motion

**How it works:**
```python
qpos = mocap.data_config[i]  # Read frame
sim.set_state(qpos, qvel)     # Directly set state
render()                      # Display
```

**Pros:**
- ✅ Perfect accuracy
- ✅ Always stable
- ✅ Fast execution

**Cons:**
- ❌ Not physically realistic (teleporting joints)
- ❌ Cannot be used for learning control policies
- ❌ No force/torque information

**Run it:**
```bash
./run_video.sh
# or
cd src && xvfb-run python dp_env_v3.py
```

---

## 2. Torque Control (`env_torque_test.py`)

**What it does:**
- Uses PD controller to compute torques
- Physics simulation drives the motion
- Attempts to track mocap reference

**How it works:**
```python
target = mocap.data_config[i]           # Read reference
current = sim.data.qpos                 # Get current state
action = Kp * (target - current)        # PD control
sim.step(action)                        # Apply torques, simulate physics
render()                                # Display
```

**Pros:**
- ✅ Physically realistic
- ✅ Uses actual motor control
- ✅ Can be used for RL training
- ✅ Shows how well control works

**Cons:**
- ❌ Tracking errors possible
- ❌ May become unstable
- ❌ Slower (physics computation)

**Run it:**
```bash
./run_torque_test.sh
# or
cd src && xvfb-run python env_torque_test.py
```

---

## Visual Comparison

### Kinematic Playback:
```
Mocap Frame 1 → Set State → Render
Mocap Frame 2 → Set State → Render
Mocap Frame 3 → Set State → Render
...
```
**Result:** Exact copy of mocap motion

### Torque Control:
```
Mocap Frame 1 → Compute Error → Calculate Torque → Simulate Physics → Render
Mocap Frame 2 → Compute Error → Calculate Torque → Simulate Physics → Render
Mocap Frame 3 → Compute Error → Calculate Torque → Simulate Physics → Render
...
```
**Result:** Approximation of mocap motion (may have tracking errors)

---

## Use Cases

### Use Kinematic Playback when:
- Visualizing mocap data
- Creating reference videos
- Debugging mocap data issues
- Need perfect reproduction

### Use Torque Control when:
- Testing control algorithms
- Training RL policies
- Evaluating controller performance
- Need physically realistic simulation

---

## Parameters

### Kinematic Playback (`dp_env_v3.py`):
```python
time_limit = 60.0  # Duration in seconds
fps = 30          # Video frame rate
width = 640       # Video width
height = 480      # Video height
```

### Torque Control (`env_torque_test.py`):
```python
time_limit = 60.0                        # Duration in seconds
Kp = 0.8                                 # Proportional gain
# ac = 0.8 * (target - current)         # PD controller
# ac += 0.02 * (target_vel - curr_vel) # Add derivative term (optional)
```

**Tuning Tips:**
- Increase `Kp` (0.8 → 1.5) for tighter tracking
- Add derivative term for smoother motion
- Reduce `Kp` (0.8 → 0.5) if motion is unstable

---

## Comparison Table

| Feature | Kinematic | Torque Control |
|---------|-----------|----------------|
| **Accuracy** | Perfect | ~95% (depends on controller) |
| **Physics** | No | Yes |
| **Stability** | Always stable | Can be unstable |
| **Speed** | Fast | Slower |
| **RL Training** | No | Yes |
| **Torque Info** | No | Yes |
| **Use Case** | Visualization | Control research |

---

## Example Output

### Kinematic Playback:
```
Starting simulation with 60.0 second time limit...
Progress: 1.0s / 60.0s - Frames: 30
Progress: 2.0s / 60.0s - Frames: 60
...
Video saved successfully!
```

### Torque Control:
```
Starting torque control test with 60.0 second time limit...
Control mode: PD controller tracking mocap reference
Progress: 1.0s / 60.0s - Frames: 30 - Reward: 0.856
Progress: 2.0s / 60.0s - Frames: 60 - Reward: 0.842
...
Video saved successfully!
```

The reward shows how well the controller is tracking the reference!

---

## Next Steps

After comparing the two approaches:

1. **Watch both videos** - Compare kinematic vs torque control
2. **Tune PD gains** - Improve tracking performance
3. **Train RL policy** - Use TRPO/GAIL to learn better control
4. **Analyze tracking errors** - Identify difficult motions

For RL training, see `trpo.py` and `gail.py`!
