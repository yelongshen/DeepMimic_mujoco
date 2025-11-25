# Gymnasium Migration Update

## Additional Migration: Gym → Gymnasium

After the initial MuJoCo migration, we discovered that the `gym` package is also deprecated and unmaintained since 2022. We've now migrated to `gymnasium`, the maintained drop-in replacement.

## What Changed

### Package Update
- **Old:** `gym>=0.18.0`
- **New:** `gymnasium>=0.29.0`

### Import Changes
The code now uses a compatibility layer that tries `gymnasium` first, falling back to `gym` if needed:

```python
try:
    import gymnasium as gym
    from gymnasium import spaces
    from gymnasium import utils
except ImportError:
    import gym
    from gym import spaces
    from gym import utils
```

### Environment Base Class
Instead of inheriting from the old `gym.envs.mujoco.MujocoEnv`, the environment now:
1. Inherits directly from `gym.Env`
2. Implements required methods manually
3. Uses our `mujoco_py_compat` wrapper for MuJoCo interactions

## Installation

Update your installation to use `gymnasium`:

```bash
pip uninstall gym
pip install gymnasium
```

Or use the updated requirements.txt:

```bash
pip install -r requirements.txt
```

## Benefits

1. **Maintained Package:** Active development and bug fixes
2. **NumPy 2.0 Support:** Works with latest NumPy versions
3. **Better API:** Improved and cleaner interfaces
4. **Future-proof:** Will receive ongoing updates

## Backward Compatibility

The code maintains backward compatibility:
- If `gymnasium` is not installed, it falls back to `gym`
- Most Gym code works without changes in Gymnasium
- API is very similar between the two

## Updated Files

### Modified for Gymnasium
- `src/dp_env_v3.py` - Now uses Gymnasium with custom Env implementation
- `requirements.txt` - Updated to use `gymnasium`
- `README.md` - Updated installation instructions

### Still Need Updates (if used)
The following files still reference `gym` and may need similar updates if you use them:
- `src/dp_env_v1.py`
- `src/dp_env_v2.py`
- `src/dp_env_test.py`
- `src/env/humanoid3d_env.py`
- `src/env/deepmimic_env_mujoco.py`
- `src/bench/monitor.py`
- `src/distributions.py`

## Migration Guide

### For End Users
Simply install `gymnasium` instead of `gym`:
```bash
pip uninstall gym
pip install gymnasium
```

### For Developers
If you're modifying the code:

1. **Replace imports:**
   ```python
   # Old
   import gym
   from gym import spaces, utils
   
   # New (with fallback)
   try:
       import gymnasium as gym
       from gymnasium import spaces, utils
   except ImportError:
       import gym
       from gym import spaces, utils
   ```

2. **Update environment class:**
   Instead of inheriting from `gym.envs.mujoco.MujocoEnv`, inherit from `gym.Env` and implement:
   - `__init__()` - Initialize observation_space and action_space
   - `step()` - Execute one timestep
   - `reset()` - Reset environment
   - `render()` - Render the environment
   - `_get_obs()` - Get observations

3. **Define spaces explicitly:**
   ```python
   self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape)
   self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
   ```

## Troubleshooting

### Error: "Gym has been unmaintained since 2022"
**Solution:** Uninstall `gym` and install `gymnasium`:
```bash
pip uninstall gym
pip install gymnasium
```

### Error: "MujocoEnv.__init__() missing 1 required positional argument"
**Solution:** This error occurs with old Gym. The updated code no longer inherits from `MujocoEnv` but implements `gym.Env` directly.

### Error: "Import gymnasium could not be resolved"
**Solution:** Install gymnasium:
```bash
pip install gymnasium
```

## Testing

Test the migration:
```bash
cd src
python dp_env_v3.py
```

You should see the humanoid animation without any deprecation warnings.

## Resources

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Migration Guide from Gym to Gymnasium](https://gymnasium.farama.org/introduction/migration_guide/)
- [Gymnasium GitHub](https://github.com/Farama-Foundation/Gymnasium)

## Summary

| Component | Old | New |
|-----------|-----|-----|
| Package | gym | gymnasium |
| Base Class | gym.envs.mujoco.MujocoEnv | gym.Env (custom implementation) |
| MuJoCo Backend | mujoco-py | mujoco + mujoco_py_compat |
| Python Support | ≤3.11 | 3.8-3.12+ |
| Status | Unmaintained | Actively maintained |

---

**Status:** Migration to Gymnasium complete for `dp_env_v3.py`. Other environment files can be updated similarly if needed.
