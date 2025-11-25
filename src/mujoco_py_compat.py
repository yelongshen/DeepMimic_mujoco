"""
Compatibility layer to bridge mujoco-py API to modern mujoco API.
This allows existing code to work with minimal changes.
"""
from __future__ import absolute_import
import sys
import os

# CRITICAL FIX: Directly import from site-packages to avoid local mujoco/ directory
# Get site-packages path and import mujoco from there explicitly
import site

_original_path = sys.path.copy()
_original_modules = sys.modules.copy()

# Remove 'mujoco' from modules if it was already imported (incorrectly)
if 'mujoco' in sys.modules:
    del sys.modules['mujoco']

# Get site-packages directories
site_packages = site.getsitepackages()
if hasattr(site, 'getusersitepackages'):
    site_packages.append(site.getusersitepackages())

# Find the real mujoco package in site-packages
mujoco = None
mujoco_path = None
import_errors = []

# Keep Python's stdlib paths (these don't contain 'site-packages' or our local directory)
stdlib_paths = [p for p in _original_path if 'site-packages' not in p and not p.startswith('/mnt/')]

for sp_dir in site_packages:
    potential_mujoco = os.path.join(sp_dir, 'mujoco')
    if os.path.isdir(potential_mujoco) and os.path.isfile(os.path.join(potential_mujoco, '__init__.py')):
        # Found a mujoco directory, try to import it
        # Set sys.path to stdlib + this site-packages (but not src directory)
        sys.path = stdlib_paths + [sp_dir]
        try:
            import mujoco as _test_mujoco
            if hasattr(_test_mujoco, 'MjModel'):
                # This is the real mujoco package!
                mujoco = _test_mujoco
                mujoco_path = potential_mujoco
                break
            else:
                # Wrong package, remove from modules
                import_errors.append(f"{potential_mujoco}: No MjModel attribute")
                if 'mujoco' in sys.modules:
                    del sys.modules['mujoco']
        except Exception as e:
            import_errors.append(f"{potential_mujoco}: {type(e).__name__}: {e}")
            if 'mujoco' in sys.modules:
                del sys.modules['mujoco']

# Restore sys.path
sys.path = _original_path

if mujoco is None:
    print("\n" + "="*70)
    print("ERROR: Could not import the correct 'mujoco' package!")
    print("="*70)
    print("\nSearched in these locations:")
    for sp_dir in site_packages:
        potential = os.path.join(sp_dir, 'mujoco')
        exists = "EXISTS" if os.path.isdir(potential) else "NOT FOUND"
        print(f"  [{exists}] {potential}")
    
    if import_errors:
        print("\nImport attempts and errors:")
        for err in import_errors:
            print(f"  - {err}")
    
    print("\n" + "="*70)
    print("WORKAROUND: Renaming local mujoco directory")
    print("="*70)
    print("\nThe local 'mujoco/' directory is conflicting.")
    print("Recommended fix:")
    print("  cd /mnt/c/DeepMimic_mujoco/src")
    print("  mv mujoco mujoco_utils")
    print("  # Then update imports in files to use 'mujoco_utils'")
    print("\nOr run from parent directory:")
    print("  cd /mnt/c/DeepMimic_mujoco")
    print("  python -m src.dp_env_v3")
    print("="*70 + "\n")
    raise ImportError("Could not import mujoco package with MjModel. See error message above.")

# Success! Now we have the real mujoco module
print(f"✓ Successfully imported mujoco {mujoco.__version__} from: {mujoco_path}")

import numpy as np
from typing import Union
import glfw


def load_model_from_xml(xml_string):
    """
    Load a MuJoCo model from an XML string.
    Compatible with mujoco-py's load_model_from_xml function.
    """
    return mujoco.MjModel.from_xml_string(xml_string)


def load_model_from_path(xml_path):
    """
    Load a MuJoCo model from an XML file path.
    Compatible with mujoco-py's load_model_from_path function.
    """
    return mujoco.MjModel.from_xml_path(xml_path)


class MjSim:
    """
    Compatibility wrapper for mujoco-py's MjSim class.
    Wraps the modern mujoco MjModel and MjData.
    """
    
    def __init__(self, model):
        """
        Initialize simulation from a MuJoCo model.
        
        Args:
            model: Either a mujoco.MjModel or a path/XML string
        """
        if isinstance(model, str):
            # If it's a string, try to load it as a file path or XML
            try:
                self.model = mujoco.MjModel.from_xml_path(model)
            except:
                self.model = mujoco.MjModel.from_xml_string(model)
        else:
            self.model = model
            
        self.data = mujoco.MjData(self.model)
        self._viewer = None
        
    def step(self):
        """Advance the simulation by one step."""
        mujoco.mj_step(self.model, self.data)
        
    def forward(self):
        """Run the forward dynamics (compute accelerations from forces)."""
        mujoco.mj_forward(self.model, self.data)
        
    def reset(self):
        """Reset the simulation to the initial state."""
        mujoco.mj_resetData(self.model, self.data)
        
    def get_state(self):
        """
        Get the full simulation state.
        Returns a flattened array of qpos and qvel.
        """
        return np.concatenate([self.data.qpos, self.data.qvel])
    
    def set_state(self, state):
        """
        Set the full simulation state from a flattened array.
        
        Args:
            state: Flattened array containing qpos and qvel
        """
        nq = self.model.nq
        self.data.qpos[:] = state[:nq]
        self.data.qvel[:] = state[nq:]
        mujoco.mj_forward(self.model, self.data)
        
    def set_state_from_flattened(self, state):
        """Alias for set_state for compatibility."""
        self.set_state(state)
        
    def render(self, width=640, height=480, camera_name=None, mode='offscreen'):
        """
        Render the simulation.
        
        Args:
            width: Width of the rendered image
            height: Height of the rendered image
            camera_name: Name of the camera to use
            mode: 'offscreen' for pixel array, 'window' for display
            
        Returns:
            numpy array of pixels if mode='offscreen', None otherwise
        """
        if mode == 'window':
            if self._viewer is None:
                self._viewer = MjViewer(self)
            self._viewer.render()
            return None
        else:
            # Offscreen rendering
            renderer = mujoco.Renderer(self.model, height=height, width=width)
            # If camera_name is None, use default camera (-1)
            if camera_name is None:
                camera_id = -1  # Default free camera
            elif isinstance(camera_name, str):
                camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            else:
                camera_id = camera_name
            renderer.update_scene(self.data, camera=camera_id)
            pixels = renderer.render()
            renderer.close()
            return pixels


class MjViewer:
    """
    Compatibility wrapper for mujoco-py's MjViewer class.
    Provides visualization using GLFW and modern mujoco rendering.
    """
    
    def __init__(self, sim):
        """
        Initialize viewer for a simulation.
        
        Args:
            sim: MjSim instance
        """
        self.sim = sim
        self.model = sim.model
        self.data = sim.data
        
        # Initialize GLFW
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
            
        # Create window
        self.window = glfw.create_window(1200, 900, "MuJoCo Viewer", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
            
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        
        # Create MuJoCo visualization structures
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        
        # Camera and interaction
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        
        # Set default camera
        self.cam.azimuth = 90
        self.cam.elevation = -20
        self.cam.distance = 5
        self.cam.lookat[0] = 0
        self.cam.lookat[1] = 0
        self.cam.lookat[2] = 1
        
        # Mouse interaction state
        self._button_left = False
        self._button_right = False
        self._last_mouse_x = 0
        self._last_mouse_y = 0
        
        # Set up callbacks
        glfw.set_cursor_pos_callback(self.window, self._mouse_move_callback)
        glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)
        
    def render(self):
        """Render one frame of the simulation."""
        if glfw.window_should_close(self.window):
            self.close()
            return
            
        # Get window size
        width, height = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, width, height)
        
        # Update scene
        mujoco.mjv_updateScene(
            self.model, self.data, self.opt, None, 
            self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scene
        )
        
        # Render scene
        mujoco.mjr_render(viewport, self.scene, self.context)
        
        # Swap buffers and poll events
        glfw.swap_buffers(self.window)
        glfw.poll_events()
        
    def close(self):
        """Close the viewer window."""
        if self.window:
            glfw.destroy_window(self.window)
            glfw.terminate()
            self.window = None
            
    def _mouse_button_callback(self, window, button, act, mods):
        """Handle mouse button events."""
        self._button_left = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
        self._button_right = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)
        
        x, y = glfw.get_cursor_pos(window)
        self._last_mouse_x = x
        self._last_mouse_y = y
        
    def _mouse_move_callback(self, window, xpos, ypos):
        """Handle mouse movement for camera control."""
        if not (self._button_left or self._button_right):
            return
            
        dx = xpos - self._last_mouse_x
        dy = ypos - self._last_mouse_y
        self._last_mouse_x = xpos
        self._last_mouse_y = ypos
        
        width, height = glfw.get_window_size(window)
        
        if self._button_left:
            # Rotate camera
            self.cam.azimuth += dx * 0.5
            self.cam.elevation += dy * 0.5
        elif self._button_right:
            # Translate camera
            self.cam.lookat[0] += dx * 0.01
            self.cam.lookat[1] -= dy * 0.01
            
    def _scroll_callback(self, window, xoffset, yoffset):
        """Handle scroll events for camera zoom."""
        self.cam.distance += yoffset * 0.2
        self.cam.distance = max(0.1, self.cam.distance)  # Prevent negative distance
        
    def __del__(self):
        """Cleanup when viewer is destroyed."""
        self.close()


class MjViewerBasic:
    """
    A simpler viewer implementation that doesn't require interactive controls.
    Useful for basic visualization without full interaction.
    """
    
    def __init__(self, sim):
        """Initialize basic viewer."""
        self.sim = sim
        self.model = sim.model
        self.data = sim.data
        
        # Initialize GLFW
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
            
        # Create window
        self.window = glfw.create_window(800, 600, "MuJoCo", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create window")
            
        glfw.make_context_current(self.window)
        
        # Create rendering context
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.camera = mujoco.MjvCamera()
        self.option = mujoco.MjvOption()
        
        # Set default camera
        mujoco.mjv_defaultCamera(self.camera)
        mujoco.mjv_defaultOption(self.option)
        
    def render(self):
        """Render the current frame."""
        if glfw.window_should_close(self.window):
            return
            
        viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)
        
        # Update scene and render
        mujoco.mjv_updateScene(self.model, self.data, self.option, None, 
                              self.camera, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        mujoco.mjr_render(viewport, self.scene, self.context)
        
        glfw.swap_buffers(self.window)
        glfw.poll_events()
        
    def close(self):
        """Close the viewer."""
        glfw.destroy_window(self.window)
        glfw.terminate()


# Module-level exports for drop-in replacement
__all__ = ['load_model_from_xml', 'load_model_from_path', 'MjSim', 'MjViewer', 'MjViewerBasic']
