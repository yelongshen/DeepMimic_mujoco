#!/usr/bin/env python3
"""
Diagnostic script to check mujoco installation and import paths.
Run this to diagnose the import issue.
"""
import sys
import os

print("="*70)
print("MuJoCo Installation Diagnostic")
print("="*70)
print()

print("Python Information:")
print(f"  Python version: {sys.version}")
print(f"  Python executable: {sys.executable}")
print(f"  Virtual env: {sys.prefix}")
print()

print("Current directory:")
print(f"  {os.getcwd()}")
print()

print("Python import paths (sys.path):")
for i, path in enumerate(sys.path[:10], 1):
    print(f"  {i}. {path}")
print()

print("Checking for 'mujoco' in each path:")
for path in sys.path[:10]:
    mujoco_path = os.path.join(path, 'mujoco')
    if os.path.exists(mujoco_path):
        is_package = os.path.isfile(os.path.join(mujoco_path, '__init__.py'))
        is_dir = os.path.isdir(mujoco_path)
        print(f"  ✓ Found at: {mujoco_path}")
        print(f"    - Is directory: {is_dir}")
        print(f"    - Is package: {is_package}")
        if is_package:
            # Try to check if it's the real mujoco
            try:
                sys.path.insert(0, path)
                import mujoco as m
                has_mjmodel = hasattr(m, 'MjModel')
                version = getattr(m, '__version__', 'unknown')
                print(f"    - Has MjModel: {has_mjmodel}")
                print(f"    - Version: {version}")
                del sys.modules['mujoco']
                sys.path.pop(0)
            except:
                print(f"    - Could not import from this location")
print()

print("Attempting to import mujoco:")
try:
    import mujoco
    print(f"  ✓ Import successful!")
    print(f"  - Module file: {mujoco.__file__}")
    print(f"  - Has MjModel: {hasattr(mujoco, 'MjModel')}")
    print(f"  - Version: {getattr(mujoco, '__version__', 'unknown')}")
    print(f"  - Dir contents: {[x for x in dir(mujoco) if not x.startswith('_')][:10]}")
except ImportError as e:
    print(f"  ✗ Import failed: {e}")
print()

print("Checking pip list for mujoco:")
import subprocess
try:
    result = subprocess.run(['pip', 'list'], capture_output=True, text=True)
    mujoco_lines = [line for line in result.stdout.split('\n') if 'mujoco' in line.lower()]
    if mujoco_lines:
        print("  Installed mujoco packages:")
        for line in mujoco_lines:
            print(f"    {line}")
    else:
        print("  ✗ No mujoco packages found in pip list")
except Exception as e:
    print(f"  Could not run pip list: {e}")
print()

print("="*70)
print("RECOMMENDATION:")
print("="*70)

# Check if local mujoco directory exists
local_mujoco = os.path.join(os.path.dirname(__file__), 'mujoco')
if os.path.exists(local_mujoco):
    print("⚠️  Local 'mujoco/' directory found in src/")
    print("   This might be conflicting with the mujoco package import.")
    print()

print("To fix this issue, run:")
print("  pip install mujoco")
print()
print("Then verify with:")
print("  python -c \"import mujoco; print(mujoco.__version__, hasattr(mujoco, 'MjModel'))\"")
print()
print("Expected output: '3.x.x True'")
print("="*70)
