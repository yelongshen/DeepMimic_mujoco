#!/usr/bin/env python
"""
Wrapper script to run TRPO evaluation with CUDA completely disabled.
This prevents TensorFlow from attempting to initialize CUDA drivers.
"""
import os
import sys

# Must be set BEFORE any TensorFlow import
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Hide CUDA libraries from TensorFlow
cuda_lib_paths = [
    '/usr/lib/wsl/lib',
    '/usr/lib/wsl/drivers',
    '/usr/local/cuda',
]

# Filter out CUDA paths from LD_LIBRARY_PATH
ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
filtered_paths = [p for p in ld_library_path.split(':') if not any(cuda in p for cuda in cuda_lib_paths)]
os.environ['LD_LIBRARY_PATH'] = ':'.join(filtered_paths)

print("Running with CUDA completely disabled (CPU-only mode)")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'not set')}")

# Now run the actual script
if __name__ == '__main__':
    # Import and run trpo after environment is set up
    sys.argv = [
        'trpo.py',
        '--task', 'evaluate',
        '--load_model_path', 'checkpoint_tmp/DeepMimic/trpo-walk-0/DeepMimic/trpo-walk-0'
    ]
    
    # Execute the trpo module
    with open('trpo.py', 'r') as f:
        code = compile(f.read(), 'trpo.py', 'exec')
        exec(code, {'__name__': '__main__', '__file__': 'trpo.py'})
