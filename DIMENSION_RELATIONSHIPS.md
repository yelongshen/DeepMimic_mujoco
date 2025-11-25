# DeepMimic MuJoCo: Dimension Relationships

## Summary of Dimensions

Based on the actual code in `dp_env_v3.py`:

```
qpos (position state):     35 dimensions  (model.nq)
qvel (velocity state):     34 dimensions  (model.nv)
action (control signals):  28 dimensions  (action_space)
observation (what AI sees): 56 dimensions  (observation_space)
```

## Detailed Breakdown

### 1. **qpos (Position State): 35 dimensions**
Full state of the humanoid skeleton positions:
- **First 7 dimensions**: Root joint (base/pelvis)
  - `[0:3]` = root position (x, y, z) in world coordinates
  - `[3:7]` = root orientation (quaternion: w, x, y, z)
- **Remaining 28 dimensions**: Body joint angles `[7:35]`
  - Chest, neck, shoulders, elbows, hips, knees, ankles, etc.
  - Total: 7 (root) + 28 (joints) = **35 dimensions**

### 2. **qvel (Velocity State): 34 dimensions**
Full state of the humanoid velocities:
- **First 6 dimensions**: Root joint velocities
  - `[0:3]` = root linear velocity (vx, vy, vz)
  - `[3:6]` = root angular velocity (wx, wy, wz)
  - Note: Orientation uses angular velocity (3D), not quaternion velocity (4D)
- **Remaining 28 dimensions**: Joint angular velocities `[6:34]`
  - Velocities for all body joints
  - Total: 6 (root) + 28 (joints) = **34 dimensions**

### 3. **Action Space: 28 dimensions**
Control signals sent to actuators:
- Each dimension controls one joint actuator
- Maps to the 28 body joints (NOT the root)
- Actions are typically in range [-1, 1] or based on `actuator_ctrlrange`
- **Root is NOT controllable** - it moves as a result of forces from other joints

### 4. **Observation Space: 56 dimensions**
What the AI policy actually sees:

```python
# From dp_env_v3.py line 188-189:
position = qpos[7:]   # 28 dims (ignore root position/orientation)
velocity = qvel[6:]   # 28 dims (ignore root linear/angular velocity)
observation = concatenate(position, velocity)  # 28 + 28 = 56 dims
```

**Why exclude root from observation?**
- Makes the policy translation-invariant (works anywhere in the world)
- Root motion is implicit in the relative joint positions
- Reduces observation complexity

## The Complete Control Loop

```
┌─────────────────────────────────────────────────────────────┐
│                     TRAINING ITERATION                       │
└─────────────────────────────────────────────────────────────┘

1. POLICY RECEIVES OBSERVATION (56 dims)
   ├─ Joint positions [28 dims] = qpos[7:]
   └─ Joint velocities [28 dims] = qvel[6:]

2. POLICY OUTPUTS ACTIONS (28 dims)
   └─ Target control for each actuator

3. MUJOCO SIMULATOR APPLIES ACTIONS
   ├─ Actions (28) → Actuators (28)
   ├─ Actuators generate torques/forces
   └─ Physics engine computes forward dynamics

4. STATE UPDATES (via physics simulation)
   ├─ Forces → Update qpos (35 dims)
   │   ├─ Root position/orientation (7 dims) - affected by physics
   │   └─ Joint angles (28 dims) - directly actuated
   ├─ Forces → Update qvel (34 dims)
   │   ├─ Root velocities (6 dims) - affected by physics
   │   └─ Joint velocities (28 dims) - directly actuated

5. NEW OBSERVATION EXTRACTED (56 dims)
   └─ qpos[7:] + qvel[6:] → observation

6. REWARD COMPUTED
   └─ Compare current pose to reference mocap frame

7. REPEAT
```

## Mathematical Relationships

### qpos Dimensions
```
qpos[0:7]   = root (position: xyz=3, orientation: quaternion=4)
qpos[7:35]  = 28 joint angles
---
Total: 35 dimensions
```

### qvel Dimensions  
```
qvel[0:6]   = root (linear velocity: xyz=3, angular velocity: xyz=3)
qvel[6:34]  = 28 joint angular velocities
---
Total: 34 dimensions (note: 34 not 35, because angular velocity is 3D not 4D)
```

### Observation Construction
```python
obs = np.concatenate([
    qpos[7:],   # 28 joint positions (exclude root 7)
    qvel[6:]    # 28 joint velocities (exclude root 6)
])
# Result: 56 dimensions
```

### Action → State Mapping
```
action[0:28] → actuator_ctrl[0:28] → torques → physics
                                              ↓
                     ┌─────────────────────────┼─────────────────┐
                     ↓                         ↓                 ↓
                qpos[7:35]                  qvel[6:34]        qpos[0:7]
              (joint angles)            (joint velocities)  qvel[0:6]
               directly                    directly         (root moves
               actuated                    actuated        due to physics)
```

## Why Different Dimensions?

**Q: Why is qpos 35 but qvel 34?**
- qpos uses **quaternion** (4D) for root orientation
- qvel uses **angular velocity** (3D) for root rotation rate
- Quaternions need 4 numbers, angular velocity only needs 3
- This is standard in robotics: orientations are 4D, rotational velocities are 3D

**Q: Why is observation 56 not 69 (35 + 34)?**
- We exclude root state from observation:
  - Removes 7 dims from qpos (root position + orientation)
  - Removes 6 dims from qvel (root linear + angular velocity)
- Observation = (35-7) + (34-6) = 28 + 28 = 56

**Q: Why are actions 28 not 35?**
- You **cannot directly control the root** (pelvis)
- Root moves as a consequence of other joint movements and physics
- You can only control the 28 body joints (arms, legs, torso, etc.)

## Humanoid Joint Breakdown (28 controllable joints)

Based on `DOF_DEF` in the code:

| Body Part          | Degrees of Freedom | Cumulative |
|--------------------|--------------------|------------|
| Chest              | 3                  | 3          |
| Neck               | 3                  | 6          |
| Right Shoulder     | 3                  | 9          |
| Right Elbow        | 1                  | 10         |
| Left Shoulder      | 3                  | 13         |
| Left Elbow         | 1                  | 14         |
| Right Hip          | 3                  | 17         |
| Right Knee         | 1                  | 18         |
| Right Ankle        | 3                  | 21         |
| Left Hip           | 3                  | 24         |
| Left Knee          | 1                  | 25         |
| Left Ankle         | 3                  | 28         |
| **Total**          | **28**             | **28**     |

## Key Takeaways

1. **State Space (Full)**: 35 (qpos) + 34 (qvel) = 69 total state dimensions
2. **Observation Space (Partial)**: 28 (joint pos) + 28 (joint vel) = 56 dimensions
3. **Action Space**: 28 dimensions (one per controllable joint)
4. **Root joint is special**:
   - Not directly controllable (no actions for it)
   - Not in observations (makes policy location-invariant)
   - Moves implicitly due to physics from other joints

5. **The policy's job**: 
   - Input: Current pose (56 dims)
   - Output: Joint targets (28 dims)
   - Goal: Match the reference mocap motion

This design is standard in humanoid control - you control the joints, and the root (center of mass) moves naturally as a result of the physics simulation.
