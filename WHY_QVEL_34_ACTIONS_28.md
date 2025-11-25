# Why qvel is 34 but actions are 28?

## Quick Answer

**qvel (34 dimensions)** = State variable (describes current motion)
**actions (28 dimensions)** = Control variable (what you can actuate)

```
qvel[0:6]   = Root velocities (6D) - PASSIVE (not controlled)
qvel[6:34]  = Joint velocities (28D) - ACTIVE (controllable)
---
Total: 34 dimensions

actions[0:28] = Joint actuator controls (28D) - matches controllable joints
```

**Key insight:** qvel includes the root, but you cannot directly control the root!

---

## Detailed Explanation

### Part 1: What is qvel?

**qvel = Generalized velocities** (state description, not control)

```python
qvel[0:6]  = Root (pelvis/base) velocities
    [0:3]  = Linear velocity (vx, vy, vz) - how fast root moves in space
    [3:6]  = Angular velocity (wx, wy, wz) - how fast root rotates

qvel[6:34] = Joint angular velocities (28 values)
    [6]    = Chest joint 1 velocity
    [7]    = Chest joint 2 velocity
    ...
    [33]   = Left ankle joint 3 velocity
```

**qvel describes the current motion state** - it's what IS happening, not what you're commanding.

### Part 2: What are actions?

**actions = Control signals** (what you command)

```python
actions[0:28] = Actuator control values
    [0]  = Chest motor 1 target
    [1]  = Chest motor 2 target
    ...
    [27] = Left ankle motor 3 target
```

**actions are commands to motors/actuators** - it's what you WANT to happen.

---

## Why the 6-Dimension Difference?

### The Root is NOT Actuated!

```
In MuJoCo XML file:

<body name="root" pos="0 0 0">
  <!-- Root is FREE-FLOATING -->
  <!-- NO actuators attached to root! -->
</body>

<body name="chest" parent="root">
  <joint name="chest_x" type="hinge"/>
  <joint name="chest_y" type="hinge"/>
  <joint name="chest_z" type="hinge"/>
</body>

<actuator>
  <!-- Motors for chest joints -->
  <motor name="chest_x_motor" joint="chest_x" gear="100"/>
  <motor name="chest_y_motor" joint="chest_y" gear="100"/>
  <motor name="chest_z_motor" joint="chest_z" gear="100"/>
  
  <!-- NO MOTORS FOR ROOT! -->
  <!-- Root moves via physics -->
</actuator>
```

**Result:**
- Root has 6 velocity dimensions in qvel (vx, vy, vz, wx, wy, wz)
- Root has 0 actuators → 0 action dimensions
- Total actions = 28 (for the 28 controllable joints)

---

## The Physics Behind It

### How Root Velocity Changes (Without Direct Control)

Even though you don't control root velocity directly, it still changes! How?

```
1. You apply actions to joint motors (28 values)
   actions[0:28] → motor torques → joint forces

2. Physics engine computes forces on root:
   - Gravity pulls down on root
   - Ground contact pushes up on root
   - Joint forces propagate to root (Newton's 3rd law)
   - Air resistance (if modeled)

3. Physics solver integrates forces → updates qvel[0:6]
   Forces → Accelerations → Velocity changes → qvel[0:6] updated

4. New root velocity affects motion
   qvel[0:6] → integrates to qpos[0:7] (root position/orientation)
```

**Example: Walking**
```
Action: Move right leg forward (action[hip] = +0.5)
  ↓
Joint motor applies torque to hip
  ↓
Leg swings forward (qvel[hip] increases)
  ↓
Ground reaction force pushes on foot
  ↓
Force propagates through leg to pelvis (root)
  ↓
Root velocity changes (qvel[0:3] increases forward)
  ↓
Body moves forward!
```

You never said "root, move forward", but it happened anyway!

---

## Comparison Table

| Property | qpos | qvel | actions |
|----------|------|------|---------|
| **Dimension** | 35 | 34 | 28 |
| **Root included?** | ✅ Yes (7D) | ✅ Yes (6D) | ❌ No (0D) |
| **Joint included?** | ✅ Yes (28D) | ✅ Yes (28D) | ✅ Yes (28D) |
| **Type** | State | State | Control |
| **Who sets it?** | Physics | Physics | Policy |
| **Root part** | Passive | Passive | N/A |
| **Joint part** | Active | Active | Active |

---

## Why This Design Makes Sense

### 1. Physical Realism
```
Real humans/robots:
  ✅ Can control joint angles (motors in joints)
  ❌ Cannot teleport (no root position control)
  ❌ Cannot fly (no root velocity control, unless you have a jetpack!)

MuJoCo mimics reality:
  ✅ Control joints via actuators (28 actions)
  ❌ No direct root control
  ✅ Root moves via physics
```

### 2. State vs Control Separation
```
State variables (qpos, qvel):
  - Describe the current situation
  - Include everything (even uncontrollable parts)
  - Updated by physics engine
  - qvel has 34 dimensions (full state)

Control variables (actions):
  - Describe your commands
  - Only include controllable parts
  - Set by policy
  - actions has 28 dimensions (controllable only)
```

### 3. Underactuated System
This is called an **underactuated system**:
```
State dimensions: 35 (qpos) + 34 (qvel) = 69 total
Control dimensions: 28 actions

69 > 28  →  Underactuated!
```

**Definition:** You have fewer controls than state dimensions.

**Consequence:** 
- You cannot independently set all state variables
- Must use physics to achieve desired root motion
- More realistic and challenging!

---

## What if You Could Control Root Velocity?

### Hypothetical: 34 Actions (Including Root)

```python
actions[0:6]  = Root velocity commands (hypothetical)
    [0:3] = Desired (vx, vy, vz)
    [3:6] = Desired (wx, wy, wz)
actions[6:34] = Joint actuator controls (28)
```

**What would happen?**

```python
# In simulation step:
if use_root_control:  # Hypothetical cheat mode
    qvel[0:6] = actions[0:6]  # Directly set root velocity
    # Apply joint actuators normally
    sim.step()
```

**Problems:**
1. ❌ **Breaks physics** - ignoring forces/momentum
2. ❌ **Unrealistic** - like having a teleport button
3. ❌ **Too easy** - no challenge in learning balance
4. ❌ **Won't transfer** - can't deploy to real robot
5. ❌ **Defeats purpose** - not learning physics-based control

**When is this used?**
- Kinematic control (animation, not physics)
- Debugging/testing
- Some video games (not simulation)

---

## Real-World Example: You!

**Your body's state (analogous to qvel):**
```
- Head velocity (uncontrolled, result of neck motion)
- Torso velocity (uncontrolled, result of leg/hip motion)
- Center of mass velocity (uncontrolled, emerges from all motions)
- Arm angular velocity (controlled by shoulder/elbow muscles)
- Leg angular velocity (controlled by hip/knee/ankle muscles)
```

**Your control (analogous to actions):**
```
- Shoulder muscle activations
- Elbow muscle activations
- Hip muscle activations
- Knee muscle activations
- Ankle muscle activations
- ... (only muscles!)
```

**You cannot directly control:**
- ❌ "Move my center of mass 2 m/s forward"
- ❌ "Rotate my torso at 10 deg/s"

**You can only control:**
- ✅ "Contract quadriceps muscle"
- ✅ "Activate calf muscles"
- ✅ These create forces → physics → you move forward

Same principle in MuJoCo!

---

## The Math

### State Space (qvel)
```
qvel ∈ ℝ³⁴  (34-dimensional state)
```

### Action Space
```
actions ∈ ℝ²⁸  (28-dimensional control)
```

### Dynamics Equation
```
qvel[6:34] influenced by: actions[0:28] (direct control)
qvel[0:6]  influenced by: physics(qpos, qvel, actions, gravity, contacts)
                          (indirect, emergent)
```

### System Type
```
State dim: 69 (35 qpos + 34 qvel)
Control dim: 28 (actions)

69 > 28  →  Underactuated system
             (Cannot independently control all states)
```

---

## Summary

| Question | Answer |
|----------|--------|
| **Why is qvel 34?** | It describes ALL velocities (6 root + 28 joints) |
| **Why are actions 28?** | You can only actuate the 28 joints, not the root |
| **Where are the missing 6?** | Root velocities (vx,vy,vz,wx,wy,wz) - uncontrollable! |
| **How does root move then?** | Physics computes it based on joint forces + gravity + contacts |
| **Is this realistic?** | ✅ Yes! Just like real humans/robots |
| **Could we add 6 root actions?** | Technically yes, but defeats the purpose (unrealistic) |

**The key insight:**

```
qvel = "What IS happening" (state)       → 34 dimensions
actions = "What you COMMAND" (control)   → 28 dimensions

The 6 missing dimensions are the root velocities,
which you cannot command directly in a physics simulation!
```

The root moves as a **consequence** of your joint actuations, not as a direct command. This is what makes physics-based control challenging and realistic!
