#!/usr/bin/env python3
"""
Test MuJoCo rendering backends for headless servers.
This script tests EGL, OSMesa, and GLFW rendering.
"""

import os
import sys

# Fix sys.path to import mujoco package instead of local directory
def fix_mujoco_import():
    """Remove local mujoco directory from sys.path to allow package import."""
    import site
    
    # Get site-packages directories
    site_packages = site.getsitepackages()
    
    # Create new path that excludes src directory but includes site-packages
    new_path = []
    src_dir = os.path.dirname(os.path.abspath(__file__))
    
    for path in sys.path:
        # Keep stdlib and site-packages, but exclude src directory
        if path == src_dir or path.startswith(src_dir):
            continue
        new_path.append(path)
    
    # Add site-packages explicitly at the front
    for sp in site_packages:
        if sp not in new_path:
            new_path.insert(0, sp)
    
    sys.path = new_path

# Fix import path before importing mujoco
fix_mujoco_import()

def test_rendering_backend(backend):
    """Test a specific rendering backend."""
    print(f"\n{'='*60}")
    print(f"Testing {backend.upper()} rendering backend")
    print('='*60)
    
    # Set the backend
    os.environ['MUJOCO_GL'] = backend
    
    try:
        # Import after setting environment variable
        import mujoco
        print(f"✓ MuJoCo imported successfully with {backend}")
        print(f"  MuJoCo version: {mujoco.__version__}")
        print(f"  MuJoCo path: {mujoco.__file__}")
        
        # Create a simple model
        xml = """
        <mujoco>
            <worldbody>
                <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
                <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
                <body pos="0 0 1">
                    <joint type="free"/>
                    <geom type="sphere" size=".1" rgba="0 .9 0 1"/>
                </body>
            </worldbody>
        </mujoco>
        """
        
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        print("✓ Model created successfully")
        
        # Test rendering
        try:
            renderer = mujoco.Renderer(model, height=480, width=640)
            print("✓ Renderer initialized")
            
            renderer.update_scene(data)
            pixels = renderer.render()
            print(f"✓ Rendered frame: shape={pixels.shape}, dtype={pixels.dtype}")
            
            renderer.close()
            print(f"✓ {backend.upper()} rendering works!")
            return True
            
        except Exception as e:
            print(f"✗ Rendering failed: {e}")
            return False
            
    except Exception as e:
        print(f"✗ Failed to initialize {backend}: {e}")
        return False

def main():
    print("MuJoCo Headless Rendering Test")
    print("="*60)
    
    # Check DISPLAY
    display = os.environ.get('DISPLAY', '')
    print(f"DISPLAY environment variable: '{display}'")
    if not display:
        print("Running in headless mode (no DISPLAY)")
    
    results = {}
    
    # Test EGL
    results['egl'] = test_rendering_backend('egl')
    
    # Test OSMesa
    results['osmesa'] = test_rendering_backend('osmesa')
    
    # Test GLFW (only if DISPLAY is available)
    if display:
        results['glfw'] = test_rendering_backend('glfw')
    else:
        print("\n" + "="*60)
        print("Skipping GLFW test (no DISPLAY available)")
        print("="*60)
        results['glfw'] = None
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for backend, success in results.items():
        if success is None:
            status = "SKIPPED"
        elif success:
            status = "✓ WORKS"
        else:
            status = "✗ FAILED"
        print(f"{backend.upper():10} : {status}")
    
    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    if results['egl']:
        print("✓ Use EGL for best performance (hardware-accelerated)")
        print("  export MUJOCO_GL=egl")
    elif results['osmesa']:
        print("✓ Use OSMesa (software rendering, slower but works)")
        print("  export MUJOCO_GL=osmesa")
    else:
        print("✗ No headless rendering backend available!")
        print("\nInstall required libraries:")
        print("\nFor EGL (recommended):")
        print("  sudo apt-get install -y libegl1-mesa libegl1-mesa-dev libgl1-mesa-glx")
        print("\nFor OSMesa (fallback):")
        print("  sudo apt-get install -y libosmesa6 libosmesa6-dev")
        print("\nOr use Xvfb:")
        print("  sudo apt-get install -y xvfb")
        print("  xvfb-run python your_script.py")

if __name__ == "__main__":
    main()
