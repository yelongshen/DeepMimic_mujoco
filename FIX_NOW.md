# Installation Steps - Fix Current Error

## Your Current Error
```
ImportError: Found a 'mujoco' module but it's not the MuJoCo physics package. 
Please install: pip install mujoco
```

## Root Cause
You're in the `openai` virtual environment, but the `mujoco` package is not installed there yet.

## Solution (Step-by-Step)

### Step 1: Make sure you're in the right virtual environment
```bash
# You should see (openai) in your prompt
# If not, activate it:
source ~/.virtualenvs/openai/bin/activate
```

### Step 2: Install the required packages
```bash
pip install mujoco glfw gymnasium numpy tensorflow pyquaternion joblib
```

**OR** use the requirements file:
```bash
cd /mnt/c/DeepMimic_mujoco
pip install -r requirements.txt
```

### Step 3: Verify installation
```bash
python -c "import mujoco; print('✓ MuJoCo installed:', mujoco.__version__)"
```

### Step 4: Run your script
```bash
cd /mnt/c/DeepMimic_mujoco/src
python dp_env_v3.py
```

## Quick One-Liner (Copy-Paste)

```bash
pip install mujoco glfw gymnasium numpy && cd /mnt/c/DeepMimic_mujoco/src && python dp_env_v3.py
```

## What We Fixed

1. **✅ Cython compilation errors** - Migrated from mujoco-py to modern mujoco
2. **✅ Gym deprecation warnings** - Migrated to gymnasium
3. **✅ Module import conflicts** - Fixed sys.path to avoid local directory conflicts
4. **⚠️ Missing installation** - You just need to install the packages now!

## Expected Output After Installation

When you run `python dp_env_v3.py`, you should see:
- A window opens showing the humanoid character
- No error messages
- Animation playing smoothly

## If Installation Fails

### Problem: pip not found or wrong pip
```bash
# Use python -m pip instead
python -m pip install mujoco glfw gymnasium numpy
```

### Problem: Permission denied
```bash
# Install to user directory
pip install --user mujoco glfw gymnasium numpy
```

### Problem: Network/proxy issues
```bash
# Try with a different index
pip install --index-url https://pypi.org/simple/ mujoco glfw gymnasium numpy
```

## Verify Everything is Ready

Run this diagnostic:
```bash
python << 'EOF'
import sys
print("Python:", sys.version)
print("Virtual env:", sys.prefix)
print()

packages = {
    'mujoco': 'MuJoCo physics engine',
    'glfw': 'Graphics library',
    'gymnasium': 'RL environment framework',
    'numpy': 'Numerical computing'
}

for pkg, desc in packages.items():
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✓ {pkg:12} {version:10} - {desc}")
    except ImportError:
        print(f"✗ {pkg:12} {'MISSING':10} - {desc} - Run: pip install {pkg}")
EOF
```

Expected output (all ✓):
```
Python: 3.12.x
Virtual env: /home/dev/.virtualenvs/openai

✓ mujoco       3.x.x      - MuJoCo physics engine
✓ glfw         2.x.x      - Graphics library
✓ gymnasium    0.29.x     - RL environment framework
✓ numpy        1.x.x      - Numerical computing
```

## Still Having Issues?

See detailed troubleshooting guides:
- **[INSTALL_MUJOCO.md](INSTALL_MUJOCO.md)** - Detailed installation instructions
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Common problems and solutions
- **[QUICKSTART.md](QUICKSTART.md)** - Complete setup guide

---

## Summary

**What to do right now:**
```bash
# 1. Install packages
pip install mujoco glfw gymnasium numpy

# 2. Test it works
python -c "import mujoco; print('Ready!')"

# 3. Run the demo
cd /mnt/c/DeepMimic_mujoco/src && python dp_env_v3.py
```

That's it! 🚀
