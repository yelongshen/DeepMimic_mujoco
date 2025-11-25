# PyTorch TRPO Implementation 🔥

This is a PyTorch implementation of Trust Region Policy Optimization (TRPO) for the DeepMimic motion imitation task, converted from the TensorFlow version.

## Key Features

✅ **Pure PyTorch** - No TensorFlow dependencies
✅ **Modern Architecture** - Uses PyTorch best practices
✅ **GPU/CPU Support** - Automatically detects and uses available hardware
✅ **Video Recording** - Built-in video capture during evaluation
✅ **Checkpoint System** - Save and load models easily
✅ **Detailed Logging** - Progress tracking with statistics

## Files

- `trpo_torch.py` - Main PyTorch TRPO implementation
- `mlp_policy_torch.py` - PyTorch MLP policy network
- `run_trpo_torch.sh` - Helper script to run with proper environment

## Installation

### Required Packages

```bash
pip install torch torchvision
pip install numpy tqdm
pip install opencv-python
pip install pyquaternion
pip install mujoco
pip install gymnasium
```

### Optional (for multi-process training)

```bash
pip install mpi4py
```

## Usage

### Training

Train a new policy from scratch:

```bash
cd /mnt/c/DeepMimic_mujoco/src
chmod +x run_trpo_torch.sh

# Train for 5M timesteps
./run_trpo_torch.sh --task train --num_timesteps 5000000 --seed 0
```

Or manually:

```bash
cd /mnt/c/DeepMimic_mujoco/src
export CUDA_VISIBLE_DEVICES=""  # Force CPU
export MUJOCO_GL="glfw"

xvfb-run -a python trpo_torch.py \
  --task train \
  --num_timesteps 5000000 \
  --save_per_iter 100 \
  --seed 0
```

### Evaluation

Evaluate a trained policy:

```bash
./run_trpo_torch.sh \
  --task evaluate \
  --load_model_path checkpoint_torch/trpo-walk-0/iter_1000.pt \
  --num_eval_episodes 10 \
  --stochastic_policy
```

### GPU Training

To use GPU (if available):

```bash
# Don't set CUDA_VISIBLE_DEVICES
python trpo_torch.py --task train --num_timesteps 5000000
```

To force CPU:

```bash
python trpo_torch.py --task train --num_timesteps 5000000 --cpu
```

## Command Line Arguments

### General

- `--task` - Task to run: 'train' or 'evaluate' (default: train)
- `--seed` - Random seed (default: 0)
- `--env_id` - Environment ID (default: DeepMimic)

### Training

- `--num_timesteps` - Total training timesteps (default: 5000000)
- `--save_per_iter` - Save checkpoint every N iterations (default: 100)
- `--policy_hidden_size` - Hidden layer size for policy (default: 100)
- `--max_kl` - Maximum KL divergence for TRPO (default: 0.01)
- `--policy_entcoeff` - Entropy coefficient (default: 0)
- `--checkpoint_dir` - Directory to save checkpoints (default: checkpoint_torch)
- `--pretrained_weight_path` - Path to pretrained weights

### Evaluation

- `--load_model_path` - Path to checkpoint file (.pt)
- `--num_eval_episodes` - Number of evaluation episodes (default: 10)
- `--max_episode_steps` - Max steps per episode (default: 2048)
- `--stochastic_policy` - Use stochastic policy (default: False/deterministic)
- `--save_sample` - Save evaluation trajectories (default: False)

### Hardware

- `--cpu` - Force CPU usage even if GPU available

## Architecture

### TRPO Algorithm

The implementation follows the standard TRPO algorithm:

1. **Collect Trajectories** - Sample from current policy
2. **Compute Advantages** - Use GAE (Generalized Advantage Estimation)
3. **Update Policy** - Use natural gradient with conjugate gradient solver
4. **Line Search** - Backtracking line search to satisfy KL constraint
5. **Update Value Function** - Train value network with Adam optimizer

### Policy Network

From `mlp_policy_torch.py`:

```
Input (Observation)
  ↓
Observation Normalization (Running Mean/Std)
  ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Policy Network          Value Network
  ↓                        ↓
Hidden Layer (tanh)     Hidden Layer (tanh)
  ↓                        ↓
Hidden Layer (tanh)     Hidden Layer (tanh)
  ↓                        ↓
Mean + LogStd           Value (scalar)
  ↓
Action Distribution (Gaussian)
  ↓
Sample Action
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Key Components

1. **Conjugate Gradient** - Efficient computation of natural gradient
2. **Fisher Vector Product** - KL divergence Hessian-vector product
3. **Line Search** - Ensures monotonic improvement
4. **GAE** - Variance reduction for advantage estimation

## Differences from TensorFlow Version

### Advantages of PyTorch Version

✅ **Cleaner Code** - More Pythonic, easier to read
✅ **Dynamic Graphs** - More flexible for debugging
✅ **Better GPU Support** - Automatic GPU detection
✅ **Modern API** - Uses current PyTorch best practices
✅ **No TF Compatibility Issues** - No need for TF1/TF2 compatibility layer

### Implementation Details

- **No Placeholders** - PyTorch uses dynamic computation graphs
- **Explicit Device Management** - Clear CPU/GPU tensor placement
- **DataLoader** - Uses PyTorch DataLoader for value function training
- **Model Checkpoints** - Saves `.pt` files instead of TensorFlow checkpoints
- **Simplified Structure** - Removed TensorFlow-specific utilities

## Checkpoint Format

PyTorch checkpoints are saved as `.pt` files containing the model's `state_dict`:

```python
# Save
torch.save(policy.state_dict(), 'checkpoint.pt')

# Load
policy.load_state_dict(torch.load('checkpoint.pt'))
```

Structure:
```
checkpoint_torch/
  └── trpo-walk-0/
      ├── iter_100.pt
      ├── iter_200.pt
      └── iter_300.pt
```

## Output

### Training Output

```
Using device: cpu
Observation space: Box(-inf, inf, (56,), float64)
Action space: Box(-0.5, 0.5, (28,), float32)

============================================================
Starting TRPO Training
Task: trpo-walk-0
Max timesteps: 5000000
============================================================

Iteration 10 | Timesteps 20480
  Avg episode length: 125.34
  Avg episode reward: 12.4523
  Policy loss: -0.0234
  KL divergence: 0.008234
  Entropy: 1.2345
  Value loss: 0.1234

💾 Saved checkpoint to checkpoint_torch/trpo-walk-0/iter_100.pt
...
```

### Evaluation Output

```
Loading checkpoint from checkpoint_torch/trpo-walk-0/iter_1000.pt
Recording video to: ./render/eval_walk_20251121_143022.avi
Evaluating 10 episodes...
100%|███████████████████| 10/10 [00:45<00:00,  4.52s/it]

============================================================
📊 EVALUATION RESULTS
============================================================
  Average episode length: 245.20 ± 12.34
  Average return:         45.6789 ± 5.6789
  Min return:             38.1234
  Max return:             55.6789
  Video saved to: ./render/eval_walk_20251121_143022.avi
============================================================
```

## Performance Tips

### Training Speed

1. **Use GPU** - 5-10x faster than CPU
   ```bash
   python trpo_torch.py --task train  # Auto-detects GPU
   ```

2. **Batch Size** - Adjust `timesteps_per_batch` in code
   - Larger = more stable but slower
   - Default: 2048

3. **Value Function Iterations** - Reduce for faster training
   - Default: 3 iterations

### Memory Usage

- **Reduce Hidden Size** - `--policy_hidden_size 64` (instead of 100)
- **Smaller Batches** - Modify `timesteps_per_batch` in code
- **Gradient Checkpointing** - Can be added for very large networks

## Troubleshooting

### Import Errors

```
ModuleNotFoundError: No module named 'torch'
```
→ Install PyTorch: `pip install torch`

### CUDA Errors

```
RuntimeError: CUDA out of memory
```
→ Use CPU mode: `--cpu` flag

### Video Recording Issues

```
libEGL warning: failed to open /dev/dri/renderD128
```
→ Use xvfb: `xvfb-run -a python trpo_torch.py ...`

### Checkpoint Loading

```
RuntimeError: Error(s) in loading state_dict
```
→ Make sure policy architecture matches checkpoint

## Comparison: TensorFlow vs PyTorch

| Feature | TensorFlow (`trpo.py`) | PyTorch (`trpo_torch.py`) |
|---------|------------------------|---------------------------|
| **Graph Type** | Static (TF1) | Dynamic |
| **Code Style** | Verbose placeholders | Clean, Pythonic |
| **GPU Support** | Manual config | Automatic |
| **Debugging** | Difficult (sessions) | Easy (native Python) |
| **Checkpoint** | Directory + meta files | Single `.pt` file |
| **Dependencies** | TF, MPI, custom utils | PyTorch, standard libs |
| **Performance** | Similar | Similar |

## Migration Guide

### Converting TF Checkpoints to PyTorch

Currently, you need to **retrain** from scratch with PyTorch. Direct checkpoint conversion is not trivial due to different parameter organization.

### Using Pretrained Weights

If you have pretrained PyTorch weights:

```bash
python trpo_torch.py \
  --task train \
  --pretrained_weight_path path/to/weights.pt \
  --num_timesteps 1000000
```

## Example Workflow

### Full Training Pipeline

```bash
# 1. Train policy
./run_trpo_torch.sh --task train --num_timesteps 5000000 --seed 0

# 2. Evaluate deterministic policy
./run_trpo_torch.sh \
  --task evaluate \
  --load_model_path checkpoint_torch/trpo-walk-0/iter_1000.pt \
  --num_eval_episodes 10

# 3. Evaluate stochastic policy
./run_trpo_torch.sh \
  --task evaluate \
  --load_model_path checkpoint_torch/trpo-walk-0/iter_1000.pt \
  --num_eval_episodes 10 \
  --stochastic_policy

# 4. Watch the videos
vlc ./render/eval_walk_*.avi
```

## Contributing

Feel free to:
- Report bugs
- Suggest improvements
- Add features (multi-agent, distributed training, etc.)
- Optimize performance

## License

Same as the original DeepMimic project.

## References

- [Trust Region Policy Optimization (TRPO)](https://arxiv.org/abs/1502.05477)
- [Generalized Advantage Estimation (GAE)](https://arxiv.org/abs/1506.02438)
- [DeepMimic Paper](https://arxiv.org/abs/1804.02717)

---

**Happy Training! 🚀**
