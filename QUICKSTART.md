# Quick Start Guide - DeepMimic with Modern MuJoCo

This guide will help you get DeepMimic running with Python 3.12+ and the modern MuJoCo package.

## Prerequisites

- Python 3.8 or later (including Python 3.12)
- pip (Python package manager)
- (Optional) Virtual environment tool

## Installation Steps

### 1. Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 2. Install Dependencies

**Quick method:**
```bash
pip install -r requirements.txt
```

**Or use the install script (Linux/macOS):**
```bash
chmod +x install.sh
./install.sh
```

**Or install manually:**
```bash
pip install mujoco glfw numpy gym tensorflow pyquaternion joblib
```

### 3. Install MPI (Optional - for parallel training)

**Ubuntu/Debian:**
```bash
sudo apt-get install openmpi-bin openmpi-common openssh-client libopenmpi-dev
pip install mpi4py
```

**macOS:**
```bash
brew install open-mpi
pip install mpi4py
```

**Windows:**
- Download Microsoft MPI from https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi
- Then: `pip install mpi4py`

## Running Examples

### Test MoCap Playback

```bash
cd src
python3 dp_env_v3.py
```

This will play back motion capture data using the physics simulator.

### Test Environment with Torque Control

```bash
cd src
python3 env_torque_test.py
```

### Train a Policy

```bash
cd src
python3 trpo.py
```

### Evaluate a Trained Policy

```bash
cd src
python3 trpo.py --task evaluate --load_model_path checkpoint_tmp/DeepMimic/trpo-walk-0/DeepMimic/trpo-walk-0
```

## Common Issues and Solutions

### Issue 1: Cython Compilation Errors
```
Error compiling Cython file... Cannot assign type 'void (const char *) except * nogil'
```

**Solution:** This was the original problem with `mujoco-py`. The migration to the modern `mujoco` package fixes this. Make sure you've installed `mujoco` (not `mujoco-py`):
```bash
pip uninstall mujoco-py
pip install mujoco glfw
```

### Issue 2: Module Import Errors
```
ModuleNotFoundError: No module named 'mujoco_py_compat'
```

**Solution:** Make sure you're running Python from the `src/` directory:
```bash
cd src
python3 dp_env_v3.py
```

Or add the src directory to your PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Issue 3: GLFW Initialization Failed
```
RuntimeError: Failed to initialize GLFW
```

**Solution:** Install GLFW system libraries:

**Ubuntu/Debian:**
```bash
sudo apt-get install libglfw3 libglfw3-dev
```

**macOS:**
```bash
brew install glfw
```

**Windows:** GLFW should be included with the pip package, but if issues persist, try:
```bash
pip install --upgrade glfw
```

### Issue 4: Display Issues on Headless Servers
If running on a server without a display:

**Option 1:** Use Xvfb (virtual framebuffer)
```bash
sudo apt-get install xvfb
xvfb-run -a python3 dp_env_v3.py
```

**Option 2:** Use offscreen rendering
Modify the viewer creation to use offscreen mode.

## Verify Installation

Run this quick test to verify everything is working:

```bash
cd src
python3 -c "from mujoco_py_compat import load_model_from_xml, MjSim, MjViewer; print('✓ MuJoCo compatibility layer working!')"
```

If this prints the success message without errors, you're ready to go!

## Next Steps

1. **Explore examples:** Try running the different demo scripts in `src/`
2. **Train your own policy:** Modify `dp_env_v3.py` to set up custom rewards
3. **Read the documentation:** See `MIGRATION_GUIDE.md` for technical details
4. **Check the original DeepMimic paper:** https://xbpeng.github.io/projects/DeepMimic/index.html

## Performance Tips

1. **Use GPU acceleration:** Make sure tensorflow-gpu is installed and working
2. **Parallel training:** Use MPI for distributed training across multiple cores
3. **Adjust simulation frequency:** Modify frame_skip in the environment for faster/slower simulation

## Getting Help

If you encounter issues:

1. Check `MIGRATION_GUIDE.md` for detailed migration information
2. Review the MuJoCo documentation: https://mujoco.readthedocs.io/
3. Check the original DeepMimic project: https://github.com/xbpeng/DeepMimic
4. Open an issue in this repository

## What Changed from Original DeepMimic?

- **MuJoCo Backend:** Uses modern `mujoco` package instead of deprecated `mujoco-py`
- **Python 3.12+ Support:** Now works with latest Python versions
- **Simplified Installation:** No need to manually download MuJoCo binaries
- **Better Compatibility:** Works on modern systems without C++ compilation issues

The core algorithms and physics remain the same - only the MuJoCo interface has been updated.

---

**Ready to get started? Run:**
```bash
cd src && python3 dp_env_v3.py
```

Happy training! 🚀
