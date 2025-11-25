# Installation Verification Checklist

Use this checklist to verify that the migration to the modern MuJoCo package was successful.

## Pre-Installation Checks

- [ ] Python version 3.8 or later is installed
  ```bash
  python3 --version
  ```

- [ ] pip is up to date
  ```bash
  python3 -m pip install --upgrade pip
  ```

- [ ] (Recommended) Virtual environment is created and activated
  ```bash
  python3 -m venv venv
  source venv/bin/activate  # Linux/macOS
  # or: venv\Scripts\activate  # Windows
  ```

## Installation Steps

- [ ] Modern MuJoCo package is installed
  ```bash
  pip install mujoco
  ```

- [ ] GLFW for visualization is installed
  ```bash
  pip install glfw
  ```

- [ ] Other dependencies are installed
  ```bash
  pip install -r requirements.txt
  ```
  Or individually:
  ```bash
  pip install numpy gym tensorflow pyquaternion joblib
  ```

- [ ] (Optional) MPI for parallel training is installed
  ```bash
  # System package first (Ubuntu/Debian):
  sudo apt-get install openmpi-bin openmpi-common openssh-client libopenmpi-dev
  
  # Then Python package:
  pip install mpi4py
  ```

## Verification Tests

### Test 1: Import Check
- [ ] Compatibility layer imports successfully
  ```bash
  cd src
  python3 -c "from mujoco_py_compat import load_model_from_xml, MjSim, MjViewer; print('✓ Success!')"
  ```
  Expected output: `✓ Success!`

### Test 2: MuJoCo Package Check
- [ ] MuJoCo package is correctly installed
  ```bash
  python3 -c "import mujoco; print(f'MuJoCo version: {mujoco.__version__}')"
  ```
  Expected output: MuJoCo version number (e.g., `3.0.0` or higher)

### Test 3: GLFW Check
- [ ] GLFW is installed and importable
  ```bash
  python3 -c "import glfw; print(f'GLFW version: {glfw.get_version_string()}')"
  ```
  Expected output: GLFW version string

### Test 4: NumPy Check
- [ ] NumPy is installed
  ```bash
  python3 -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
  ```

### Test 5: Run Basic Environment
- [ ] Basic environment runs without errors
  ```bash
  cd src
  python3 dp_env_v3.py
  ```
  Expected: A window should open showing the humanoid animation

### Test 6: Run Torque Test
- [ ] Torque control test runs
  ```bash
  cd src
  python3 env_torque_test.py
  ```

### Test 7: Check Training Script
- [ ] Training script loads without errors
  ```bash
  cd src
  python3 -c "import trpo; print('✓ TRPO module loaded successfully')"
  ```

## Common Issues Resolution

### Issue: Cython Compilation Errors
- [ ] **Verified:** You are NOT seeing errors like:
  ```
  Cannot assign type 'void (const char *) except * nogil'
  ```
  If you see this, you're still using `mujoco-py`. Solution:
  ```bash
  pip uninstall mujoco-py
  pip install mujoco glfw
  ```

### Issue: Import ModuleNotFoundError
- [ ] **Verified:** All imports work from the `src/` directory
  If not, ensure you're running from the correct directory:
  ```bash
  cd /path/to/DeepMimic_mujoco/src
  python3 your_script.py
  ```

### Issue: GLFW Initialization Failed
- [ ] **Verified:** GLFW initializes successfully
  If not, install system GLFW libraries:
  ```bash
  # Ubuntu/Debian:
  sudo apt-get install libglfw3 libglfw3-dev
  
  # macOS:
  brew install glfw
  ```

### Issue: Display Not Available (Headless Server)
- [ ] **If running on headless server:** Xvfb is installed
  ```bash
  sudo apt-get install xvfb
  xvfb-run -a python3 dp_env_v3.py
  ```

## Final Verification

- [ ] All files have been updated from `mujoco-py` to `mujoco_py_compat`
- [ ] Documentation files are present:
  - [ ] QUICKSTART.md
  - [ ] MIGRATION_GUIDE.md
  - [ ] MIGRATION_SUMMARY.md
  - [ ] requirements.txt
  - [ ] install.sh

- [ ] At least one demo runs successfully without errors
- [ ] Python 3.12+ works (if that was your goal)

## Success Criteria

✅ **Migration is successful if:**
1. Python 3.12 (or your target version) is being used
2. No Cython compilation errors occur
3. At least one demo script runs and displays animation
4. All imports work correctly

## Troubleshooting Resources

If any check fails, consult:
1. **QUICKSTART.md** - Common issues and quick solutions
2. **MIGRATION_GUIDE.md** - Detailed technical information
3. **MuJoCo Documentation** - https://mujoco.readthedocs.io/

## Report Issues

If you encounter issues not covered in the documentation:
1. Check MuJoCo GitHub discussions: https://github.com/deepmind/mujoco/discussions
2. Review this project's issues on GitHub
3. Ensure all dependencies are at required versions

---

## Quick Command Summary

```bash
# Full installation from scratch
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Quick test
cd src
python3 dp_env_v3.py

# Verify imports
python3 -c "from mujoco_py_compat import MjSim; print('✓ Working!')"
```

---

**Status:** [ ] All checks passed - Ready to use!

Date: _______________
Python Version: _______________
MuJoCo Version: _______________
