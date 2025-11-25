# Quick Fix for "Could not import the 'mujoco' package" Error

## The Issue
You're seeing this error because the `mujoco` Python package is not installed in your `openai` virtual environment.

## Quick Solution

```bash
# Activate your virtual environment (if not already activated)
source ~/.virtualenvs/openai/bin/activate  # Linux/macOS
# or on Windows: .virtualenvs\openai\Scripts\activate

# Install the required packages
pip install mujoco glfw gymnasium numpy
```

## Verify Installation

After installing, verify it worked:

```bash
python -c "import mujoco; print('MuJoCo version:', mujoco.__version__); print('MjModel exists:', hasattr(mujoco, 'MjModel'))"
```

Expected output:
```
MuJoCo version: 3.x.x
MjModel exists: True
```

## Then Run Your Script

```bash
cd /mnt/c/DeepMimic_mujoco/src
python dp_env_v3.py
```

## Complete Installation Command (One-Liner)

```bash
pip install mujoco glfw gymnasium numpy tensorflow pyquaternion joblib && cd /mnt/c/DeepMimic_mujoco/src && python dp_env_v3.py
```

## If You Get Permission Errors

```bash
pip install --user mujoco glfw gymnasium numpy
```

## Check What's Installed

```bash
pip list | grep -E "mujoco|glfw|gymnasium"
```

Should show:
```
gymnasium    0.29.x
glfw         2.x.x
mujoco       3.x.x
```

## Common Mistakes to Avoid

❌ **Don't install:** `pip install mujoco-py` (old, deprecated, causes Cython errors)  
✅ **Do install:** `pip install mujoco` (modern, maintained, Python 3.12 compatible)

---

**Quick Copy-Paste:**
```bash
pip install mujoco glfw gymnasium numpy && python dp_env_v3.py
```
