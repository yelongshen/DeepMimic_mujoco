#!/usr/bin/env python3
"""
Quick test to verify mujoco installation.
"""
print("Testing mujoco installation...")
print()

# Test 1: Can we import mujoco at all?
print("Test 1: Import mujoco module")
try:
    import mujoco
    print(f"  ✓ Import successful")
    print(f"    Location: {mujoco.__file__}")
except ImportError as e:
    print(f"  ✗ FAILED: {e}")
    print()
    print("ACTION REQUIRED:")
    print("  pip install mujoco")
    exit(1)

print()

# Test 2: Does it have MjModel?
print("Test 2: Check for MjModel class")
if hasattr(mujoco, 'MjModel'):
    print(f"  ✓ MjModel found")
else:
    print(f"  ✗ FAILED: MjModel not found")
    print(f"    Available attributes: {[x for x in dir(mujoco) if not x.startswith('_')][:10]}")
    print()
    print("This means you have a 'mujoco' module but it's not the MuJoCo physics package.")
    print("Likely causes:")
    print("  1. The local mujoco/ directory is being imported")
    print("  2. Wrong package is installed")
    print()
    print("ACTION REQUIRED:")
    print("  pip uninstall mujoco  # Remove any wrong package")
    print("  pip install mujoco    # Install the correct one")
    exit(1)

print()

# Test 3: Can we get the version?
print("Test 3: Check version")
try:
    version = mujoco.__version__
    print(f"  ✓ Version: {version}")
except AttributeError:
    print(f"  ⚠ Warning: No __version__ attribute (might be okay)")

print()

# Test 4: Can we create a simple model?
print("Test 4: Create a simple model")
try:
    xml = "<mujoco><worldbody><body><geom size='1'/></body></worldbody></mujoco>"
    model = mujoco.MjModel.from_xml_string(xml)
    print(f"  ✓ Model created successfully")
    print(f"    Number of bodies: {model.nbody}")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    exit(1)

print()

# Test 5: Can we import the compat layer?
print("Test 5: Import compatibility layer")
try:
    from mujoco_py_compat import MjSim, MjViewer, load_model_from_xml
    print(f"  ✓ Compatibility layer imported successfully")
except ImportError as e:
    print(f"  ✗ FAILED: {e}")
    print()
    print("The compatibility layer couldn't import.")
    print("This is the issue preventing dp_env_v3.py from running.")
    exit(1)

print()
print("="*70)
print("✓ ALL TESTS PASSED!")
print("="*70)
print()
print("Your mujoco installation is correct.")
print("You should be able to run: python dp_env_v3.py")
print()
