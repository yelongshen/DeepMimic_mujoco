# How ob_rms is Automatically Saved in PyTorch

## TL;DR: YES, ob_rms IS SAVED ✅

When you call `torch.save(policy.state_dict(), 'model.pth')`, the `ob_rms` statistics are **automatically included** because they are registered as **buffers**.

---

## Understanding PyTorch's Parameter System

### Three Types of Model State

1. **Parameters** (`nn.Parameter`) - Trainable
   - Updated by optimizer
   - Have gradients
   - Example: `self.pol_mean.weight`, `self.pol_mean.bias`

2. **Buffers** (`register_buffer()`) - Non-trainable but saved
   - NOT updated by optimizer
   - NO gradients
   - Saved in `state_dict()`
   - Example: `self.ob_rms._sum`, `self.ob_rms._count`

3. **Regular attributes** - Not saved
   - Not included in `state_dict()`
   - Lost when saving/loading
   - Example: temporary variables

---

## How ob_rms is Implemented

### In mlp_policy_torch.py (lines 20-31):

```python
class RunningMeanStd(nn.Module):
    def __init__(self, epsilon=1e-2, shape=()):
        super(RunningMeanStd, self).__init__()
        self.epsilon = epsilon
        self.shape = shape
        
        # 🔑 KEY: register_buffer makes these part of state_dict
        self.register_buffer('_sum', torch.zeros(shape, dtype=torch.float64))
        self.register_buffer('_sumsq', torch.full(shape, epsilon, dtype=torch.float64))
        self.register_buffer('_count', torch.tensor(epsilon, dtype=torch.float64))
```

### What `register_buffer()` does:

```python
# WITHOUT register_buffer:
self._sum = torch.zeros(shape)  # ❌ NOT saved in state_dict()

# WITH register_buffer:
self.register_buffer('_sum', torch.zeros(shape))  # ✅ SAVED in state_dict()
```

---

## What Gets Saved in Your Model

When you save your SFT model:

```python
torch.save(policy.state_dict(), 'policy_sft_pretrained.pth')
```

The checkpoint includes:

```
policy_sft_pretrained.pth contains:
├── ob_rms._sum          [56]  ← Observation statistics
├── ob_rms._sumsq        [56]  ← Observation statistics  
├── ob_rms._count        []    ← Observation statistics
├── vf_net.0.weight      [64, 56]
├── vf_net.0.bias        [64]
├── vf_net.2.weight      [64, 64]
├── vf_net.2.bias        [64]
├── vf_final.weight      [1, 64]
├── vf_final.bias        [1]
├── pol_net.0.weight     [64, 56]
├── pol_net.0.bias       [64]
├── pol_net.2.weight     [64, 64]
├── pol_net.2.bias       [64]
├── pol_mean.weight      [28, 64]
├── pol_mean.bias        [28]
└── pol_logstd           [1, 28]
```

---

## Why This Matters

### Scenario: Training and Inference

**Training:**
```python
# Train SFT model
policy.ob_rms.update(training_observations)  # Update statistics
# Now: mean = [5.2, 0.1, ...], std = [3.1, 2.5, ...]

torch.save(policy.state_dict(), 'model.pth')  # Save including ob_rms
```

**Inference (weeks later):**
```python
# Load model
policy_new = MlpPolicy(...)
policy_new.load_state_dict(torch.load('model.pth'))  # ob_rms restored!

# Normalization uses SAME statistics as training
obs_normalized = (obs - policy_new.ob_rms.mean) / policy_new.ob_rms.std
# ✅ Consistent normalization = better performance
```

### What Would Happen Without Saving ob_rms?

```python
# If ob_rms was NOT saved:
policy_new.ob_rms.mean  # = [0, 0, 0, ...]  (default)
policy_new.ob_rms.std   # = [1, 1, 1, ...]  (default)

# Wrong normalization!
obs_normalized = (obs - [0, 0, ...]) / [1, 1, ...]  # ❌ Different from training!
# Result: Poor performance, model doesn't work properly
```

---

## Verification Test

### Test 1: Check Checkpoint Contents

```python
import torch

ckpt = torch.load('policy_sft_pretrained.pth')

# Check if ob_rms is there
ob_rms_keys = [k for k in ckpt.keys() if 'ob_rms' in k]
print(ob_rms_keys)
# Output: ['ob_rms._sum', 'ob_rms._sumsq', 'ob_rms._count']
```

### Test 2: Save and Load Test

```python
# Create and train policy
policy1 = MlpPolicy(...)
policy1.ob_rms.update(data)  # mean becomes [5.0, 2.0, ...]

# Save
torch.save(policy1.state_dict(), 'test.pth')

# Create fresh policy
policy2 = MlpPolicy(...)  # mean is [0.0, 0.0, ...] (default)

# Load
policy2.load_state_dict(torch.load('test.pth'))
print(policy2.ob_rms.mean)  # [5.0, 2.0, ...] ✅ Restored!
```

---

## Common Pitfall: NOT UPDATING ob_rms

### Before Our Fix (in train_sft.py):

```python
# ob_rms was never updated!
# Stayed at default: mean=0, std=1
obs_normalized = (obs - 0) / 1 = obs  # No normalization!
```

### After Our Fix:

```python
# Update ob_rms before training
policy.ob_rms.update(all_training_observations)

# Now normalization actually works
obs_normalized = (obs - learned_mean) / learned_std
```

**The fix ensures:**
1. ✅ Statistics are computed from training data
2. ✅ Saved in checkpoint automatically (via `register_buffer`)
3. ✅ Loaded when reusing model
4. ✅ Consistent normalization train ↔ test

---

## Summary

| Aspect | Status |
|--------|--------|
| Is ob_rms saved? | ✅ YES (automatically via `register_buffer`) |
| Need manual code to save it? | ❌ NO (handled by PyTorch) |
| Loaded automatically? | ✅ YES (part of `load_state_dict`) |
| Moved to GPU automatically? | ✅ YES (via `model.to('cuda')`) |
| Updated by optimizer? | ❌ NO (not trainable) |
| Need to manually update? | ✅ YES (call `ob_rms.update(data)`) |

---

## Best Practices

### ✅ DO:
```python
# Update ob_rms with training data
policy.ob_rms.update(training_observations)

# Save model (ob_rms included automatically)
torch.save(policy.state_dict(), 'model.pth')

# Load model (ob_rms restored automatically)
policy.load_state_dict(torch.load('model.pth'))
```

### ❌ DON'T:
```python
# Don't manually save/load ob_rms separately
# torch.save(policy.ob_rms, 'stats.pth')  # Unnecessary!

# Don't reset ob_rms after loading
# policy.ob_rms = RunningMeanStd(...)  # This loses loaded statistics!
```

---

## Code Reference

**mlp_policy_torch.py, line 170:**
```python
# Observation normalization
self.ob_rms = RunningMeanStd(shape=ob_space.shape)
```

This `ob_rms` is an `nn.Module` with buffers, so it's automatically part of `state_dict()`.

**train_sft.py, line 136 (after fix):**
```python
# Update observation statistics for normalization
all_obs = np.array([x[0] for x in train_data])
self.policy.ob_rms.update(torch.tensor(all_obs, dtype=torch.float32))
```

This updates the statistics, which are then saved when you call `torch.save(policy.state_dict(), ...)`.

---

## Conclusion

**YES, ob_rms is automatically saved!** You don't need any special code. PyTorch's `register_buffer()` mechanism handles it for you. The only thing you need to do is **update** ob_rms with your training data before/during training.
