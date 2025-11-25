# Why Not Use Full State? (35+34 obs, 34 actions)

## Your Question
**Why not use:**
- Observation: All 35 qpos + 34 qvel = **69 dimensions**
- Actions: All 34 qvel dimensions = **34 actions**

**Current design:**
- Observation: 28 joint pos + 28 joint vel = **56 dimensions** (excludes root)
- Actions: **28 actuators** (excludes root velocity)

---

## Reason 1: Root is NOT Directly Controllable

### The Physical Reality
**You cannot directly set the root velocity in a physics simulator!**

```python
# What you CANNOT do in MuJoCo:
qvel[0:6] = desired_root_velocity  # ❌ This breaks physics!

# What you CAN do:
actuator_ctrl[:] = desired_joint_torques  # ✅ Apply forces to joints
# Then physics computes the resulting root motion
```

### Why?
- The root (pelvis/base) is a **free-floating body** in the world
- Its motion is **determined by physics** (Newton's laws)
- You can only apply forces/torques to joints
- Root motion emerges as a **consequence** of joint forces + gravity + ground contact

**Analogy:** 
- You can't directly control where your body's center of mass goes
- You can only move your legs/arms, and your body moves as a result
- Same principle here!

---

## Reason 2: Translation Invariance (Generalization)

### Problem with Including Root Position in Observation

If you include root position `qpos[0:3]` (x, y, z coordinates):

```
Training scenario:
  Dance motion at position (0, 0, 0)
  Policy learns: "When I see position (0, 0, 0), do dance move A"

Testing scenario:
  Try same dance at position (10, 5, 0)
  Policy sees: position (10, 5, 0) - completely new input!
  Result: ❌ Policy fails - it's never seen these coordinates before
```

### Solution: Exclude Root Position
```
Policy only sees relative joint angles (not world position)
  ✅ Dance move works at ANY location
  ✅ Policy is translation-invariant
  ✅ Better generalization
```

**Real-world example:**
- You can perform the same dance move anywhere in a room
- You don't need to memorize "left arm up when at x=10m"
- You just know "left arm up relative to body"

---

## Reason 3: Rotation Invariance

### Problem with Including Root Orientation

If you include root orientation `qpos[3:7]` (quaternion):

```
Training:
  Dance facing North (yaw = 0°)
  Policy learns orientation-specific patterns

Testing:
  Dance facing East (yaw = 90°)
  ❌ Policy confused - different orientation quaternion
```

### Solution: Exclude Root Orientation
```
Policy only sees:
  - Joint angles relative to torso
  - Joint velocities relative to torso
  
✅ Can perform same motion facing ANY direction
```

---

## Reason 4: Action Space Must Match Actuators

### MuJoCo's Actuator System

```python
# MuJoCo model defines actuators in XML:
<actuator>
  <motor name="chest_joint" gear="100"/>
  <motor name="neck_joint" gear="100"/>
  <motor name="right_hip" gear="100"/>
  ...
  <!-- 28 actuators total -->
</actuator>
```

**Key point:** There is NO actuator for the root!

### What Actions Actually Do

```python
# In MuJoCo simulation:
sim.data.ctrl[:] = actions  # Set actuator control signals
sim.step()                   # Physics computes resulting motion

# Internally:
# - Each action sets a motor/actuator target
# - Motors apply torques to joints
# - Physics solver computes qpos, qvel changes
# - Root moves as a consequence of joint forces
```

**You physically cannot have 34 actions** because there are only 28 actuators in the model!

---

## Reason 5: Learning Efficiency

### Information Redundancy

Root state is **redundant** - it's already implicit in joint configuration:

```
If you know:
  - All joint angles (28)
  - All joint velocities (28)
  - Ground contact forces
  
You can infer:
  - Where the root is moving
  - How the body is oriented
  - Center of mass trajectory
```

### Smaller Observation = Easier Learning

```
69-dim observation space:
  - More neurons needed
  - More training data needed
  - Harder to learn patterns
  - Slower convergence

56-dim observation space:
  - Fewer neurons needed
  - Less redundant information
  - Focus on what's controllable
  - ✅ Faster learning
```

---

## Reason 6: The Task Doesn't Need Root State

### What the Policy Actually Needs to Know

For motion imitation (matching a reference dance):

**Need to know:**
- ✅ Joint angles - to match reference pose
- ✅ Joint velocities - to match reference motion
- ✅ Relative body positions - for coordination

**Don't need to know:**
- ❌ Absolute world position (x, y, z)
- ❌ Absolute world orientation (yaw)
- ❌ Root linear velocity (vx, vy, vz)

**Why?** The reference mocap data also doesn't care about absolute position:
```python
# Reward function compares:
current_joint_angles ≈ reference_joint_angles  # ✅ Makes sense
current_joint_vels ≈ reference_joint_vels      # ✅ Makes sense

# It does NOT compare:
current_world_position ≈ reference_world_position  # ❌ Overly restrictive
```

---

## Could You Include Root State? (Theoretical Analysis)

### Scenario A: Include Root in Observation, Keep 28 Actions

```python
observation = qpos[0:35] + qvel[0:34]  # 69 dims
actions = 28 actuator controls
```

**Problems:**
1. ❌ Policy sees root state but **cannot control it directly**
   - Confusing for learning - "I see this, but I can't change it directly"
2. ❌ **Breaks translation invariance**
   - Dance only works at training location
3. ❌ **Breaks rotation invariance**
   - Dance only works at training orientation
4. ❌ **More data needed** to learn the pattern
5. ✅ Might help slightly with balance/momentum awareness

**Verdict:** Bad idea for this task

---

### Scenario B: 56 Observation, Try to Use 34 Actions

```python
observation = qpos[7:35] + qvel[6:34]  # 56 dims
actions = 34 dimensions  # ❌ But only 28 actuators exist!
```

**Problems:**
1. ❌ **Physically impossible** - only 28 actuators in the model
2. ❌ Where would actions[28:34] go?
   - Can't control root velocity directly in physics sim
3. ❌ Would need to redesign the entire MuJoCo model

**Verdict:** Physically impossible

---

### Scenario C: Full 69 Observation + Add Root Actuators (34 actions)

```python
observation = qpos[0:35] + qvel[0:34]  # 69 dims
actions = 28 joints + 6 root forces     # 34 dims

# New action space:
actions[0:28] = joint actuators
actions[28:34] = [root_fx, root_fy, root_fz, root_tx, root_ty, root_tz]
```

**Could this work?**
- ✅ Technically feasible (add 6 actuators to root in XML)
- ❌ **Unrealistic** - humans don't have rocket boosters!
- ❌ Defeats the purpose of physics-based learning
- ❌ Not true imitation - using "cheat" forces
- ❌ Won't transfer to real robots

**Verdict:** Defeats the purpose of the project

---

## What Design IS Used in Other Tasks?

### When Full Root State Makes Sense

**Task: Navigation/Locomotion in large environments**
```python
# Goal: Walk from point A to point B
observation = [
    qpos[0:35],  # Include root position - need to know where you are!
    qvel[0:34],  # Include root velocity - need to track progress!
    target_position,  # Where to go
]
actions = 28  # Still only joint actuators

# Why include root?
# - Need to know distance to goal
# - Need to track if making progress
# - Translation invariance not needed
```

**Task: Manipulation (robot arm with fixed base)**
```python
# Robot arm is fixed to table
observation = [
    qpos[7:35],   # Joint angles (no free-floating root!)
    qvel[7:35],   # Joint velocities
    object_position,  # What to manipulate
]
actions = 28

# Why no root?
# - Base is fixed (not free-floating)
# - Only joints move
```

**Task: Flying drone**
```python
observation = [
    qpos[0:7],   # Position + orientation (critical for flight!)
    qvel[0:6],   # Velocities (need for stability)
    # ... other sensors
]
actions = 4  # Four rotor speeds

# Why include root?
# - Root IS the controllable part
# - Propellers directly control root motion
```

---

## Summary: Why Current Design is Optimal

| Design Choice | Reason |
|---------------|--------|
| **Observation = 56** (exclude root) | Translation/rotation invariance, generalization, efficiency |
| **Actions = 28** (joint actuators only) | Matches physical actuators, physics-realistic control |
| **No root control** | Root motion emerges from physics (realistic!) |
| **No root observation** | Task doesn't need it, and it hurts generalization |

---

## The Deep Learning Perspective

### What Neural Networks Learn

**With root state (69 obs):**
```
Network learns: 
  "When at position (x, y, z) with orientation (q), 
   and joints at angles θ, apply actions α"

Problem: Memorizes location-specific patterns
```

**Without root state (56 obs):**
```
Network learns:
  "When joints at angles θ with velocities θ̇,
   apply actions α to match reference motion"

Benefit: Learns abstract motion patterns (works anywhere!)
```

---

## Conclusion

**Your proposal:**
- Obs: 35 + 34 = 69
- Actions: 34

**Why it doesn't work:**
1. ❌ Only 28 actuators exist (can't have 34 actions)
2. ❌ Root velocity not directly controllable in physics
3. ❌ Including root position breaks translation invariance
4. ❌ Including root orientation breaks rotation invariance
5. ❌ More dimensions = slower learning with no benefit
6. ❌ Task (motion imitation) doesn't need root state

**Current design (56 obs, 28 actions) is optimal for:**
- ✅ Physics realism
- ✅ Generalization
- ✅ Learning efficiency
- ✅ The specific task of motion imitation

The key insight: **The root is not controlled, it's a consequence!** Just like in real life - you don't control your center of mass directly, it moves as a result of how you move your limbs.
