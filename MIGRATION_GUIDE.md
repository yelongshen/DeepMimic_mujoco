# Migration from mujoco-py to mujoco

This document describes the migration from the deprecated `mujoco-py` package to the modern `mujoco` package maintained by DeepMind.

## Why Migrate?

- **Python 3.12+ Support**: `mujoco-py` has Cython compilation issues with Python 3.12+
- **Official Support**: The new `mujoco` package is officially maintained by DeepMind
- **No License Required**: MuJoCo is now free and open-source
- **Better Performance**: Modern package with improved performance and stability
- **Active Development**: Regular updates and bug fixes

## What Changed?

### Installation

**Old (mujoco-py):**
```bash
# Download MuJoCo 210 manually
mkdir -p ~/.mujoco
cd ~/.mujoco
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
tar -xzf mujoco210-linux-x86_64.tar.gz

# Install mujoco-py
pip install mujoco-py
```

**New (mujoco):**
```bash
# Everything is bundled - just install the package
pip install mujoco
pip install glfw  # For visualization
```

### Code Changes

All imports have been updated to use a compatibility wrapper (`mujoco_py_compat.py`) that bridges the old API to the new one:

**Old:**
```python
from mujoco_py import load_model_from_xml, MjSim, MjViewer
```

**New:**
```python
from mujoco_py_compat import load_model_from_xml, MjSim, MjViewer
```

### Compatibility Layer

A compatibility wrapper (`src/mujoco_py_compat.py`) has been created to minimize code changes. This wrapper:

1. **Translates API calls**: Converts old `mujoco-py` API to new `mujoco` API
2. **Maintains compatibility**: Existing code works with minimal changes
3. **Provides wrappers for**:
   - `load_model_from_xml()` - Load models from XML strings
   - `load_model_from_path()` - Load models from file paths
   - `MjSim` - Simulation wrapper class
   - `MjViewer` - Visualization wrapper class

### Files Modified

All files that imported `mujoco-py` have been updated:

**Main environment files:**
- `src/dp_env_v1.py`
- `src/dp_env_v2.py`
- `src/dp_env_v3.py`
- `src/dp_env_test.py`
- `src/play_mocap.py`

**Utility files:**
- `src/mujoco/mocap_v1.py`
- `src/mujoco/mocap_v2.py`
- `src/mujoco/mujoco_env.py`
- `src/mujoco/setting_states.py`
- `src/env/setting_states.py`

## Installation Instructions

### Step 1: Install Python 3.8+
Make sure you have Python 3.8 or later (including Python 3.12):
```bash
python3 --version
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
# Core MuJoCo package
pip install mujoco

# Visualization
pip install glfw

# Other dependencies
pip install gym numpy tensorflow-gpu pyquaternion joblib

# MPI for parallel training
sudo apt-get install openmpi-bin openmpi-common openssh-client libopenmpi-dev
pip install mpi4py
```

### Step 4: Test the Installation
```bash
cd src
python3 dp_env_v3.py
```

## Troubleshooting

### Issue: "Failed to initialize GLFW"
**Solution**: Install GLFW system library
```bash
# Ubuntu/Debian
sudo apt-get install libglfw3 libglfw3-dev

# macOS
brew install glfw

# Windows
# GLFW is included in the pip package
```

### Issue: Import errors with mujoco_py_compat
**Solution**: Make sure you're running Python from the `src/` directory or add it to your PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/DeepMimic_mujoco/src"
```

### Issue: "module 'mujoco' has no attribute 'MjModel'"
**Solution**: Make sure you installed the correct `mujoco` package:
```bash
pip uninstall mujoco mujoco-py
pip install mujoco
```

## API Differences

While the compatibility wrapper handles most differences, here are the key API changes for reference:

| mujoco-py | mujoco (new) |
|-----------|--------------|
| `MjSim(model)` | `MjData(model)` + manual step calls |
| `sim.data.qpos` | `data.qpos` (direct access) |
| `sim.step()` | `mujoco.mj_step(model, data)` |
| `sim.forward()` | `mujoco.mj_forward(model, data)` |
| `MjViewer(sim)` | Custom viewer using `mujoco.Renderer` |

The compatibility wrapper in `mujoco_py_compat.py` handles these translations automatically.

## Benefits of the New Package

1. **Simpler Installation**: No need to download MuJoCo separately
2. **Better Performance**: Optimized Python bindings
3. **Modern Python Support**: Works with Python 3.12+
4. **Active Maintenance**: Regular updates from DeepMind
5. **Better Documentation**: Official docs at https://mujoco.readthedocs.io/

## Additional Resources

- [MuJoCo Python Bindings Documentation](https://mujoco.readthedocs.io/en/stable/python.html)
- [MuJoCo GitHub Repository](https://github.com/deepmind/mujoco)
- [MuJoCo Forum](https://github.com/deepmind/mujoco/discussions)

## Rollback (If Needed)

If you need to rollback to the old `mujoco-py` setup:

1. Use Python 3.11 or earlier
2. Revert all imports from `mujoco_py_compat` back to `mujoco_py`
3. Download and install MuJoCo 210 manually
4. Install `mujoco-py` package

However, we recommend sticking with the new setup for better long-term compatibility.
