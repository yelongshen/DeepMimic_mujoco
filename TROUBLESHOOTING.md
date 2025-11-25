# Common Issues and Solutions

## Issue: "module 'mujoco' has no attribute 'MjModel'"

### Error Message
```
AttributeError: module 'mujoco' has no attribute 'MjModel'
```

### Cause
This error occurs when Python imports the local `mujoco/` directory instead of the `mujoco` package. This happens because:
1. You have a directory named `mujoco/` in the `src/` folder
2. Python searches the current directory first when importing modules
3. It finds the local `mujoco/` directory before the installed `mujoco` package

### Solution 1: The Fix is Already Applied ✅
The `mujoco_py_compat.py` file has been updated to use `importlib` to force importing the correct package:

```python
import importlib
mujoco = importlib.import_module('mujoco')  # Forces the package, not local directory
```

This should resolve the issue automatically.

### Solution 2: If the Error Persists

If you still see the error, ensure you have the `mujoco` package installed:

```bash
pip install mujoco
```

**Verify installation:**
```bash
python -c "import importlib; m = importlib.import_module('mujoco'); print('MuJoCo version:', m.__version__); print('Has MjModel:', hasattr(m, 'MjModel'))"
```

Expected output:
```
MuJoCo version: 3.x.x
Has MjModel: True
```

### Solution 3: Check for Conflicting Packages

Make sure you don't have the old `mujoco-py` installed:

```bash
pip uninstall mujoco-py
pip install mujoco
```

### Solution 4: Clear Python Cache

Sometimes Python caches old imports:

```bash
# Remove __pycache__ directories
cd src
find . -type d -name __pycache__ -exec rm -rf {} +

# Or on Windows PowerShell:
Get-ChildItem -Path . -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
```

Then try running again:
```bash
python dp_env_v3.py
```

## Issue: "Import 'mujoco' could not be resolved"

### Error Message
```
ImportError: No module named 'mujoco'
```

### Solution
Install the mujoco package:

```bash
pip install mujoco
```

## Issue: "Failed to initialize GLFW"

### Error Message
```
RuntimeError: Failed to initialize GLFW
```

### Solution
Install GLFW:

**Python package:**
```bash
pip install glfw
```

**System library (if needed):**

**Ubuntu/Debian:**
```bash
sudo apt-get install libglfw3 libglfw3-dev
```

**macOS:**
```bash
brew install glfw
```

**Windows:**
Usually the pip package is sufficient. If issues persist:
```bash
pip install --upgrade glfw
```

## Issue: "No module named 'gymnasium'"

### Error Message
```
ImportError: No module named 'gymnasium'
```

### Solution
Install gymnasium:

```bash
pip uninstall gym  # Remove old gym if present
pip install gymnasium
```

## Issue: Directory Structure Conflicts

### Problem
The local `mujoco/` directory conflicts with the `mujoco` package import.

### Why This Happens
Python's import system searches in this order:
1. Current directory and its subdirectories
2. PYTHONPATH directories
3. Standard library
4. Site-packages (where pip installs packages)

Because the local `mujoco/` directory is found first, Python imports it instead of the package.

### Our Solution
We use `importlib.import_module()` which bypasses the current directory search and goes straight to installed packages.

### Alternative Solution (Not Recommended)
You could rename the local `mujoco/` directory to something else like `mujoco_utils/`, but this would require updating many imports throughout the codebase.

## Issue: Multiple Python Versions

### Problem
You might have mujoco installed for Python 3.11 but running with Python 3.12.

### Solution
Make sure you're using the right Python and pip:

```bash
# Check Python version
python --version
python3 --version

# Install for specific Python version
python3.12 -m pip install mujoco glfw gymnasium

# Run with specific Python version
python3.12 dp_env_v3.py
```

## Issue: Virtual Environment Issues

### Problem
Packages installed globally but not in virtual environment.

### Solution
Activate your virtual environment first:

```bash
# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate

# Then install
pip install -r requirements.txt
```

## Testing Your Installation

Run this comprehensive test:

```bash
cd src
python -c "
import importlib
import sys

print('Python version:', sys.version)
print()

# Test mujoco package
try:
    mujoco = importlib.import_module('mujoco')
    print('✓ mujoco package:', mujoco.__version__)
    print('✓ Has MjModel:', hasattr(mujoco, 'MjModel'))
except Exception as e:
    print('✗ mujoco package:', e)

# Test glfw
try:
    import glfw
    print('✓ glfw:', glfw.get_version_string())
except Exception as e:
    print('✗ glfw:', e)

# Test gymnasium
try:
    import gymnasium
    print('✓ gymnasium:', gymnasium.__version__)
except Exception as e:
    print('✗ gymnasium:', e)

# Test numpy
try:
    import numpy
    print('✓ numpy:', numpy.__version__)
except Exception as e:
    print('✗ numpy:', e)

# Test compatibility wrapper
try:
    from mujoco_py_compat import MjSim, MjViewer
    print('✓ mujoco_py_compat: Working')
except Exception as e:
    print('✗ mujoco_py_compat:', e)
"
```

Expected output (all ✓):
```
Python version: 3.12.x

✓ mujoco package: 3.x.x
✓ Has MjModel: True
✓ glfw: 3.x.x
✓ gymnasium: 0.29.x
✓ numpy: 1.x.x or 2.x.x
✓ mujoco_py_compat: Working
```

## Still Having Issues?

1. **Read the documentation:**
   - [QUICKSTART.md](QUICKSTART.md)
   - [COMPLETE_MIGRATION.md](COMPLETE_MIGRATION.md)
   - [INSTALLATION_CHECKLIST.md](INSTALLATION_CHECKLIST.md)

2. **Clean install:**
   ```bash
   # Uninstall everything
   pip uninstall mujoco mujoco-py gym gymnasium glfw
   
   # Reinstall from requirements
   pip install -r requirements.txt
   ```

3. **Check your environment:**
   ```bash
   pip list | grep -E "mujoco|gym|glfw"
   ```

4. **Try in a fresh virtual environment:**
   ```bash
   python -m venv fresh_env
   source fresh_env/bin/activate  # or fresh_env\Scripts\activate on Windows
   pip install -r requirements.txt
   cd src
   python dp_env_v3.py
   ```

---

**Most Common Solution:**
```bash
pip install mujoco glfw gymnasium numpy
cd src
python dp_env_v3.py
```

This should work in 95% of cases! 🎯
