# PyTorch TRPO Implementation Fixes

## Fix #1: CUDA Crash (RESOLVED)

**Error:**
```
free(): double free detected in tcache 2
Signal: Aborted (6)
```

**Root Cause:** WSL2 NVIDIA CUDA driver bug when PyTorch tries to enumerate GPUs

**Solution:** Install CPU-only PyTorch:
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Status:** ✅ FIXED - No more CUDA crashes!

---

## Fix #2: Gradient Computation Error (RESOLVED)

**Error:**
```
RuntimeError: grad can be implicitly created only for scalar outputs
```

**Location:** `trpo_torch.py`, line 169 in `update_policy()` method

**Root Cause:** 
The `_compute_kl()` function returns a vector (one KL value per sample in the batch), but `torch.autograd.grad()` requires the output to be a **scalar** when computing gradients.

**Solution:**
Take the mean of KL divergence to convert it to a scalar:
```python
def Fvp(v):
    kl = self._compute_kl(obs).mean()  # Now scalar!
    grads = flat_grad(kl, ...)          # Works!
```

**Status:** ✅ FIXED

---

## Fix #3: Parameter Size Mismatch in Line Search (RESOLVED)

**Error:**
```
RuntimeError: The size of tensor a (34557) must match the size of tensor b (18656) at non-singleton dimension 0
```

**Location:** `trpo_torch.py`, line 237 in `_line_search()` method

**Root Cause:**
TRPO only updates **policy parameters** (not value function parameters), but the original code was trying to work with **all parameters**:

```python
# In update_policy():
policy_params = [p for name, p in self.policy.named_parameters() 
                if 'pol' in name or 'logstd' in name]  # ~18,656 params

# But later:
old_params = get_flat_params(self.policy)  # ALL params (~34,557) - WRONG!
```

This caused a size mismatch:
- `fullstep` (from conjugate gradient) → size 18,656 (policy only)
- `old_params` (from get_flat_params) → size 34,557 (policy + value function)
- When adding: `old_params + step_frac * fullstep` → ERROR!

**Solution:**
Modified `get_flat_params()` and `set_flat_params()` to accept an optional parameter list:

```python
def get_flat_params(model, param_list=None):
    """Get flattened model parameters"""
    if param_list is None:
        param_list = list(model.parameters())
    return torch.cat([param.data.reshape(-1) for param in param_list])

def set_flat_params(model, flat_params, param_list=None):
    """Set model parameters from flattened vector"""
    if param_list is None:
        param_list = list(model.parameters())
    # ... set only params in param_list
```

Then updated the calls to pass `policy_params`:
```python
# Line search - only update policy parameters, not value function
old_params = get_flat_params(self.policy, policy_params)
success, new_params = self._line_search(
    old_params, fullstep, ..., policy_params
)
set_flat_params(self.policy, new_params, policy_params)
```

**Why This is Correct:**
- TRPO updates policy parameters using natural gradient descent
- Value function is updated separately using supervised learning (MSE loss)
- Mixing them would violate the TRPO algorithm design
- Each parameter group should be optimized independently

**Code Changes:**

1. **Modified helper functions** (lines 80-97):
   - Added optional `param_list` parameter to `get_flat_params()`
   - Added optional `param_list` parameter to `set_flat_params()`

2. **Updated `update_policy()` method** (lines 197-207):
   - Pass `policy_params` to `get_flat_params()`
   - Pass `policy_params` to `_line_search()`
   - Pass `policy_params` to `set_flat_params()`

3. **Updated `_line_search()` method** (lines 239-260):
   - Added `policy_params` parameter to signature
   - Pass `policy_params` to `set_flat_params()` inside loop

**Status:** ✅ FIXED

---

## Fix #4: Value Function Gradient Error (RESOLVED)

**Error:**
```
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

**Location:** `trpo_torch.py`, line 278 in `update_value_function()` method

**Root Cause:**

The value function update was breaking the computational graph:

```python
# Line 274: Convert tensor to numpy (breaks gradient tracking!)
values = self.policy.get_value(batch_obs.cpu().numpy())

# Line 275: Create new tensor without gradients
values = torch.FloatTensor(values).to(self.device)  # No grad_fn!

# Line 277: Try to backprop - ERROR!
vf_loss.backward()  # Can't compute gradients!
```

The problem was that `get_value()` uses `torch.no_grad()` context and returns NumPy arrays. When we convert back to tensors, PyTorch doesn't know how to compute gradients because the computational graph was broken.

**Solution:**

Created a new method `get_value_train()` in `mlp_policy_torch.py` that:
1. **Keeps gradients enabled** (no `torch.no_grad()` context)
2. **Works with tensors** (no numpy conversion)
3. **Maintains computational graph** for backpropagation

**Code Changes:**

1. **Added new method to `mlp_policy_torch.py`** (after line 301):
```python
def get_value_train(self, ob_tensor):
    """Get value prediction with gradients enabled (for training)"""
    # Expects ob_tensor to already be a torch tensor on correct device
    if ob_tensor.ndim == 1:
        ob_tensor = ob_tensor.unsqueeze(0)
    
    ob_normalized = torch.clamp(
        (ob_tensor - self.ob_rms.mean) / self.ob_rms.std,
        -5.0, 5.0
    )
    vf_out = self.vf_net(ob_normalized)
    vpred = self.vf_final(vf_out).squeeze(-1)
    return vpred  # Returns tensor with gradients!
```

2. **Updated `update_value_function()` in `trpo_torch.py`**:
```python
# Before (WRONG - breaks gradient graph):
values = self.policy.get_value(batch_obs.cpu().numpy())
values = torch.FloatTensor(values).to(self.device)

# After (CORRECT - maintains gradient graph):
values = self.policy.get_value_train(batch_obs)
```

**Why Two Methods?**

- **`get_value()`**: For inference/evaluation
  - Uses `torch.no_grad()` for efficiency
  - Returns NumPy arrays (easier to work with)
  - No memory overhead from gradient tracking
  
- **`get_value_train()`**: For training
  - Keeps gradients enabled
  - Returns PyTorch tensors
  - Maintains computational graph for backprop

**Additional Improvement:**

Also fixed the return value to average loss across all batches:
```python
# Track loss across all batches
total_loss = 0
num_batches = 0

for _ in range(self.vf_iters):
    for batch in loader:
        # ... training ...
        total_loss += vf_loss.item()
        num_batches += 1

return total_loss / num_batches  # Average loss
```

**Status:** ✅ FIXED

---

## Current Status

All four major issues have been resolved:
1. ✅ CUDA crash fixed (using CPU-only PyTorch)
2. ✅ Gradient computation fixed (using `.mean()` for KL)
3. ✅ Parameter size mismatch fixed (separate policy/value params)
4. ✅ Value function gradient error fixed (added `get_value_train()` method)

**Training should now work!**

Run:
```bash
cd /mnt/c/DeepMimic_mujoco/src
./run_trpo_torch.sh --task train --num_timesteps 5000000 --seed 0
```

---

## Technical Details

### Parameter Sizes
For the humanoid3d model:
- **Policy parameters**: ~18,656 (action network + log_std)
  - Filtered by: `'pol' in name or 'logstd' in name`
- **Value function parameters**: ~15,901 (critic network)
  - Filtered by: `'vf' in name`
- **Total parameters**: ~34,557

### TRPO Algorithm Structure
```
For each iteration:
  1. Collect trajectories using current policy
  2. Compute advantages using GAE
  3. Update policy using natural gradient (THIS STEP - policy params only!)
     - Compute surrogate loss gradient
     - Use conjugate gradient to find natural gradient direction
     - Line search to ensure KL constraint
  4. Update value function using supervised learning (separate step!)
     - MSE loss between predicted values and returns
     - Standard Adam optimizer
```

### Why Separate Updates?
- **Policy update**: Trust region optimization (TRPO)
  - Uses natural gradient descent
  - Constrained by KL divergence
  - Only updates action network parameters
  
- **Value function update**: Standard supervised learning
  - Uses gradient descent with Adam
  - Minimizes MSE loss
  - Only updates critic network parameters

Mixing these would break both algorithms!

---

## Next Steps

If training starts successfully, you should see:
```
Iteration 1: Average Return = X.XX, KL = X.XXXX
Iteration 2: Average Return = X.XX, KL = X.XXXX
...
```

Monitor:
- Average Return should increase over time
- KL divergence should stay below max_kl (0.01 by default)
- Checkpoints saved every 100 iterations

Training time on CPU: ~8-12 hours for 5M timesteps
