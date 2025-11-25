# Intro
> Status: Archive (code is provided as-is)

> **✨ Update (2025):** This project has been migrated to use the modern `mujoco` package, enabling **Python 3.12+ support**! See [QUICKSTART.md](QUICKSTART.md) for installation instructions.

Mujoco version of [DeepMimic](https://xbpeng.github.io/projects/DeepMimic/index.html): 
* No C++ codes --> pure python
* No bullet engine --> Mujoco engine
* No PPO --> TRPO-based 

Examples: 

* Walk (play MoCap data):
<img src="docs/walk.gif" alt="walk" width="400px"/>

* Spinkick (play MoCap data):
<img src="docs/spinkick.gif" alt="spinkick" width="400px"/>

* Dance_b (play MoCap data):
<img src="docs/dance.gif" alt="dance" width="400px"/>

* Stand up straight (training via TRPO):
<img src="docs/standup.gif" alt="standup" width="400px"/>

# Install

**Note:** This project has been migrated to:
1. Use the modern `mujoco` Python package (officially maintained by DeepMind) which supports Python 3.8+ including Python 3.12
2. Use `gymnasium` instead of deprecated `gym` package

📚 **Documentation:**
- [QUICKSTART.md](QUICKSTART.md) - Quick installation and getting started guide
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues and solutions
- [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) - Detailed migration information  
- [MIGRATION_SUMMARY.md](MIGRATION_SUMMARY.md) - Summary of changes made

## Quick Install

### Option 1: Using the install script (Linux/macOS)
``` bash
chmod +x install.sh
./install.sh
```

### Option 2: Using requirements.txt
``` bash
pip install -r requirements.txt
```

### Option 3: Manual installation

* Mujoco: Install the modern MuJoCo Python bindings:
``` bash
# Install the modern mujoco package (includes MuJoCo binaries)
python3 -m pip install mujoco

# Install GLFW for visualization (required by the compatibility wrapper)
python3 -m pip install glfw
```

* python3 modules: python dependencies
``` bash
python3 -m pip install gymnasium
python3 -m pip install tensorflow>=2.10  # TensorFlow 2.x with TF1 compatibility
python3 -m pip install pyquaternion
python3 -m pip install joblib
python3 -m pip install numpy
```

**Note:** 
- We use `gymnasium` (the maintained replacement for `gym`)
- TensorFlow 2.x is supported via TF1 compatibility mode (see [TENSORFLOW_COMPATIBILITY.md](TENSORFLOW_COMPATIBILITY.md))
- For GPU support: `pip install tensorflow[and-cuda]>=2.10`

* MPI & MPI4PY: mpi for parrellel training
``` bash 
sudo apt-get install openmpi-bin openmpi-common openssh-client libopenmpi-dev
python3 -m pip install mpi4py
```

# Usage
* Testing examples:
``` bash
python3 dp_env_v3.py # play a mocap
python3 env_torque_test.py # torque control with p-controller
```

* Gym env

Before training a policy:
**Modify the step in dp_env_v3.py to set up correct rewards for the task**. Use **dp_env_v3.py** as the training env.

Training a policy that makes the agent stands up straight:
``` bash
python3 trpo.py
```
Running a policy:
``` bash
python3 trpo.py --task evaluate --load_model_path XXXX # for evaluation
# e.g., python3 trpo.py --task evaluate --load_model_path checkpoint_tmp/DeepMimic/trpo-walk-0/DeepMimic/trpo-walk-0
```

# Acknowledge

This repository is based on code accompanying the SIGGRAPH 2018 paper:
"DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills".
The framework uses reinforcement learning to train a simulated humanoid to imitate a variety
of motion skills from mocap data.
Project page: https://xbpeng.github.io/projects/DeepMimic/index.html
