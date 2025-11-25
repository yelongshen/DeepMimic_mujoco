"""
Alternative compatibility wrapper that uses a different approach.
Rename this to mujoco_py_compat.py if the original doesn't work.
"""
from __future__ import absolute_import
import sys
import os

# Get the site-packages directory where pip installs packages
import site
site_packages = site.getsitepackages()

# Try to import mujoco from site-packages explicitly
mujoco = None
for sp_path in site_packages:
    mujoco_path = os.path.join(sp_path, 'mujoco')
    if os.path.exists(mujoco_path) and os.path.isfile(os.path.join(mujoco_path, '__init__.py')):
        # Found mujoco in site-packages, add it first to sys.path
        if sp_path not in sys.path:
            sys.path.insert(0, sp_path)
        try:
            import mujoco as _mujoco
            if hasattr(_mujoco, 'MjModel'):
                mujoco = _mujoco
                break
        except ImportError:
            pass

if mujoco is None:
    print("\n" + "="*70)
    print("ERROR: MuJoCo package not found!")
    print("="*70)
    print("\nSearched in:")
    for sp in site_packages:
        print(f"  - {sp}")
    print("\nThe 'mujoco' package must be installed.")
    print("\nRun: pip install mujoco")
    print("\nTo verify installation:")
    print("  python -c \"import mujoco; print(mujoco.__version__)\"")
    print("="*70 + "\n")
    raise ImportError("MuJoCo package not installed. Run: pip install mujoco")

import numpy as np
from typing import Union
import glfw


def load_model_from_xml(xml_string):
    """Load a MuJoCo model from an XML string."""
    return mujoco.MjModel.from_xml_string(xml_string)


def load_model_from_path(xml_path):
    """Load a MuJoCo model from an XML file path."""
    return mujoco.MjModel.from_xml_path(xml_path)


class MjSim:
    """Compatibility wrapper for mujoco-py's MjSim class."""
    
    def __init__(self, model):
        if isinstance(model, str):
            try:
                self.model = mujoco.MjModel.from_xml_path(model)
            except:
                self.model = mujoco.MjModel.from_xml_string(model)
        else:
            self.model = model
            
        self.data = mujoco.MjData(self.model)
        self._viewer = None
        
    def step(self):
        mujoco.mj_step(self.model, self.data)
        
    def forward(self):
        mujoco.mj_forward(self.model, self.data)
        
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        
    def get_state(self):
        return np.concatenate([self.data.qpos, self.data.qvel])
    
    def set_state(self, state):
        nq = self.model.nq
        self.data.qpos[:] = state[:nq]
        self.data.qvel[:] = state[nq:]
        mujoco.mj_forward(self.model, self.data)
        
    def set_state_from_flattened(self, state):
        self.set_state(state)
        
    def render(self, width=640, height=480, camera_name=None, mode='offscreen'):
        if mode == 'window':
            if self._viewer is None:
                self._viewer = MjViewer(self)
            self._viewer.render()
            return None
        else:
            renderer = mujoco.Renderer(self.model, height=height, width=width)
            renderer.update_scene(self.data, camera=camera_name)
            pixels = renderer.render()
            return pixels


class MjViewer:
    """Compatibility wrapper for mujoco-py's MjViewer class."""
    
    def __init__(self, sim):
        self.sim = sim
        self.model = sim.model
        self.data = sim.data
        
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
            
        self.window = glfw.create_window(1200, 900, "MuJoCo Viewer", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
            
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        
        self.cam.azimuth = 90
        self.cam.elevation = -20
        self.cam.distance = 5
        self.cam.lookat[0] = 0
        self.cam.lookat[1] = 0
        self.cam.lookat[2] = 1
        
        self._button_left = False
        self._button_right = False
        self._last_mouse_x = 0
        self._last_mouse_y = 0
        
        glfw.set_cursor_pos_callback(self.window, self._mouse_move_callback)
        glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)
        
    def render(self):
        if glfw.window_should_close(self.window):
            self.close()
            return
            
        width, height = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, width, height)
        
        mujoco.mjv_updateScene(
            self.model, self.data, self.opt, None, 
            self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scene
        )
        
        mujoco.mjr_render(viewport, self.scene, self.context)
        
        glfw.swap_buffers(self.window)
        glfw.poll_events()
        
    def close(self):
        if self.window:
            glfw.destroy_window(self.window)
            glfw.terminate()
            self.window = None
            
    def _mouse_button_callback(self, window, button, act, mods):
        self._button_left = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
        self._button_right = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)
        
        x, y = glfw.get_cursor_pos(window)
        self._last_mouse_x = x
        self._last_mouse_y = y
        
    def _mouse_move_callback(self, window, xpos, ypos):
        if not (self._button_left or self._button_right):
            return
            
        dx = xpos - self._last_mouse_x
        dy = ypos - self._last_mouse_y
        self._last_mouse_x = xpos
        self._last_mouse_y = ypos
        
        if self._button_left:
            self.cam.azimuth += dx * 0.5
            self.cam.elevation += dy * 0.5
        elif self._button_right:
            self.cam.lookat[0] += dx * 0.01
            self.cam.lookat[1] -= dy * 0.01
            
    def _scroll_callback(self, window, xoffset, yoffset):
        self.cam.distance += yoffset * 0.2
        self.cam.distance = max(0.1, self.cam.distance)
        
    def __del__(self):
        self.close()


__all__ = ['load_model_from_xml', 'load_model_from_path', 'MjSim', 'MjViewer']
