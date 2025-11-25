# PyTorch CUDA Crash Fix for WSL2

## Problem

When running PyTorch TRPO training, you get this error:
```
free(): double free detected in tcache 2
[DESKTOP-xxx:xxxxx] *** Process received signal ***
[DESKTOP-xxx:xxxxx] Signal: Aborted (6)
...
/usr/lib/wsl/drivers/.../libcuda.so.1.1(+0x4f6fa5)
...
libcudart.so.12(cudaGetDeviceCount+0x4a)
libc10_cuda.so(_ZN3c104cuda12device_countEv+0x5a)
libtorch_cpu.so(_ZN2at14getAccelerator...)
```

## Root Cause

The WSL2 NVIDIA CUDA driver (`libcuda.so.1.1`) has a memory corruption bug (double-free error) that crashes when PyTorch or TensorFlow tries to enumerate CUDA devices using `cudaGetDeviceCount()`.

This is a **known WSL2 driver bug**, not a bug in your code or PyTorch.

## Why Environment Variables Don't Work

Setting `CUDA_VISIBLE_DEVICES=""` helps in many cases, but PyTorch's standard installation includes CUDA libraries that are **compiled into** the binary. Even when you hide devices, PyTorch's autograd engine (`libtorch_cpu.so`) tries to detect accelerators during the backward pass, which triggers the buggy CUDA driver.

## Solution: Install CPU-Only PyTorch

The **only reliable fix** is to install PyTorch's official CPU-only build, which is compiled without any CUDA support.

### Quick Fix (Recommended)

Run the provided script:
```bash
cd /mnt/c/DeepMimic_mujoco/src
chmod +x reinstall_pytorch_cpu.sh
./reinstall_pytorch_cpu.sh
```

### Manual Installation

If you prefer to do it manually:

1. **Activate your virtual environment:**
   ```bash
   source ~/.virtualenvs/openai/bin/activate
   ```

2. **Uninstall current PyTorch:**
   ```bash
   pip uninstall torch torchvision torchaudio
   ```

3. **Install CPU-only version:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

4. **Verify installation:**
   ```bash
   python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

   You should see:
   ```
   PyTorch 2.x.x+cpu
   CUDA available: False
   ```

## After Installation

Once CPU-only PyTorch is installed, you can run training normally:

```bash
cd /mnt/c/DeepMimic_mujoco/src
./run_trpo_torch.sh --task train --num_timesteps 5000000 --seed 0
```

## Performance Impact

**CPU-only training will be slower** than GPU training, but it's the only stable option on WSL2 with the current driver bug.

Estimated training times for 5M timesteps:
- GPU (RTX 4060): ~2-4 hours (if it worked)
- CPU (modern multi-core): ~8-12 hours

You can reduce timesteps for testing:
```bash
# Quick test with 100K timesteps (~10-20 minutes)
./run_trpo_torch.sh --task train --num_timesteps 100000 --seed 0
```

## Alternative: Use Native Linux or Docker

If you need GPU training, consider:

1. **Native Linux installation** (no WSL2)
2. **Docker container** with proper CUDA isolation
3. **Google Colab** or **cloud GPU instances**

## Why This Only Affects WSL2

- **Native Linux**: CUDA drivers work correctly
- **WSL2**: Uses a special NVIDIA driver (`/usr/lib/wsl/drivers/...`) with known memory management bugs
- **Windows**: Native Windows PyTorch with CUDA works fine

## Technical Details

The crash happens here in the stack trace:
```
torch.autograd.grad() 
  → compute_dependencies()
    → getAccelerator() 
      → cuda::device_count() 
        → cudaGetDeviceCount() 
          → libcuda.so.1.1 
            → DOUBLE FREE BUG → CRASH
```

PyTorch's CPU library (`libtorch_cpu.so`) still tries to detect CUDA availability for optimization purposes, even when you're using CPU tensors. The only way to prevent this is to use a build that was **compiled without CUDA support**.

## Verification

After installing CPU-only PyTorch, check:

```bash
# Should show CPU-only version
python -c "import torch; print(torch.__version__)"
# Expected: 2.x.x+cpu

# Should return False
python -c "import torch; print(torch.cuda.is_available())"
# Expected: False

# Should not have CUDA libs
ls ~/.virtualenvs/openai/lib/python3.12/site-packages/torch/lib/ | grep cuda
# Expected: No output (or only cuda_runtime libs, not libcudart.so)
```

## Summary

**Problem**: WSL2 CUDA driver crashes when PyTorch detects GPUs  
**Cause**: Driver bug (double-free) in `/usr/lib/wsl/drivers/.../libcuda.so.1.1`  
**Solution**: Install PyTorch CPU-only build (no CUDA compiled in)  
**Trade-off**: Slower training, but stable and functional  

Run `./reinstall_pytorch_cpu.sh` and you're good to go!
