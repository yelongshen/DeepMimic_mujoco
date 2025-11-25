# Headless Rendering Setup for MuJoCo

When running on a server without a display (headless mode), MuJoCo needs special configuration for rendering.

## Quick Fix

Try running with different rendering backends:

### Option 1: EGL (Hardware-accelerated, recommended)
```bash
export MUJOCO_GL=egl
python src/dp_env_v3.py
```

**If you get "gladLoadGL error"**, install EGL libraries:
```bash
sudo apt-get update
sudo apt-get install -y libegl1-mesa libegl1-mesa-dev libgl1-mesa-glx libgles2-mesa-dev
```

### Option 2: OSMesa (Software rendering, slower but more compatible)
```bash
# Install OSMesa
sudo apt-get update
sudo apt-get install -y libosmesa6 libosmesa6-dev

# Install MuJoCo with OSMesa support
pip install mujoco

# Run with OSMesa
export MUJOCO_GL=osmesa
python src/dp_env_v3.py
```

### Option 3: Xvfb (Virtual framebuffer)
```bash
# Install Xvfb
sudo apt-get update
sudo apt-get install -y xvfb

# Run with virtual display
xvfb-run -a -s "-screen 0 1400x900x24" python src/dp_env_v3.py
```

## Testing Rendering

Test which rendering backend works:

```bash
cd src
python test_rendering.py
```

## Current Configuration

The `dp_env_v3.py` script automatically sets `MUJOCO_GL=egl` when no DISPLAY is detected.

## Troubleshooting

### Error: "gladLoadGL error"
- **Cause**: EGL libraries not installed or GPU drivers missing
- **Solution**: Install EGL libraries (see Option 1) or use OSMesa (Option 2)

### Error: "Could not initialize GLFW"
- **Cause**: Trying to use window rendering without display
- **Solution**: Use offscreen rendering with EGL or OSMesa

### Error: "OSMesa not available"
- **Cause**: OSMesa libraries not installed
- **Solution**: Install libosmesa6-dev and libosmesa6

## Performance Comparison

- **EGL**: Fast, hardware-accelerated, requires GPU
- **OSMesa**: Slower, software-based, works everywhere
- **Xvfb**: Medium speed, requires X11 packages

## Recommended Setup for Production

For WSL2/Ubuntu headless servers:

```bash
# Install EGL (try this first)
sudo apt-get update
sudo apt-get install -y \
    libegl1-mesa \
    libegl1-mesa-dev \
    libgl1-mesa-glx \
    libgles2-mesa-dev

# If EGL doesn't work, install OSMesa as fallback
sudo apt-get install -y libosmesa6 libosmesa6-dev

# Test
export MUJOCO_GL=egl
python src/dp_env_v3.py

# If that fails, try:
export MUJOCO_GL=osmesa
python src/dp_env_v3.py
```
