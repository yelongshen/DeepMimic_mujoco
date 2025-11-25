# Why Observation Normalization is Critical

## The Problem: Different Feature Scales

### Example Raw Observation (56-dim)
```
obs = [
    # Root position (x, y, z)
    10.5, 0.85, 0.02,          # meters - scale: ~10
    
    # Root orientation (quaternion)
    1.0, 0.0, 0.0, 0.0,        # unit quaternion - scale: ~1
    
    # Joint angles (28 joints)
    0.1, -0.2, 0.5, ...,       # radians - scale: ~3
    
    # Linear velocity (3)
    2.5, 0.1, 0.0,             # m/s - scale: ~10
    
    # Angular velocity (3)
    0.5, 0.3, -0.1,            # rad/s - scale: ~5
    
    # Joint velocities (28)
    1.2, -0.5, 0.8, ...,       # rad/s - scale: ~10
]
```

### Problem in Neural Network

```python
# Layer computation: output = weights @ input + bias

# WITHOUT NORMALIZATION:
output = w1 * 10.5 + w2 * 0.85 + w3 * 0.02 + w4 * 1.0 + w5 * 0.1 + ...
         ^^^^^^^^^   ^^^^^^^^^^   ^^^^^^^^^^
         DOMINATES!  medium       tiny - ignored!
```

**Result**: Network only learns from large-magnitude features (position), ignores important small features (joint angles).

## The Solution: Standardization

### Normalize to N(0, 1)
```python
obs_normalized = (obs - mean) / std
```

All features now have:
- **Mean**: 0
- **Standard deviation**: 1

### After Normalization
```
obs_normalized = [
    # All features now ~[-3, 3] range
    0.2, -0.5, 0.1, 0.0, 0.3, -0.1, ...
]
```

```python
# WITH NORMALIZATION:
output = w1 * 0.2 + w2 * (-0.5) + w3 * 0.1 + w4 * 0.0 + w5 * 0.3 + ...
         ^^^^^^^^   ^^^^^^^^^^^^   ^^^^^^^^^   ^^^^^^^^   ^^^^^^^^^
         ALL FEATURES CONTRIBUTE EQUALLY!
```

## Implementation in Code

### 1. Running Statistics Tracker
```python
class RunningMeanStd:
    """Tracks mean and std of observations online"""
    def __init__(self, shape):
        self._sum = torch.zeros(shape)
        self._sumsq = torch.zeros(shape)
        self._count = 0
    
    def update(self, x):
        """Update with new batch of observations"""
        self._sum += x.sum(dim=0)
        self._sumsq += (x ** 2).sum(dim=0)
        self._count += x.shape[0]
    
    @property
    def mean(self):
        return self._sum / self._count
    
    @property
    def std(self):
        return sqrt(self._sumsq / self._count - self.mean ** 2)
```

### 2. During SFT Training (train_sft.py)

```python
# STEP 1: Update statistics with training data
all_obs = np.array([x[0] for x in train_data])
policy.ob_rms.update(torch.tensor(all_obs))

# STEP 2: Normalize during training
obs_normalized = (obs_batch - policy.ob_rms.mean) / policy.ob_rms.std
```

### 3. During Inference (policy.forward())

```python
def forward(self, ob):
    # Normalize and clip outliers
    ob_normalized = torch.clamp(
        (ob - self.ob_rms.mean) / self.ob_rms.std,
        -5.0, 5.0  # Prevent extreme outliers
    )
    
    # Feed to network
    output = self.pol_net(ob_normalized)
    ...
```

## Benefits

✅ **Faster Learning**: All features contribute to gradients

✅ **Better Convergence**: Optimization is easier with similar scales

✅ **Improved Stability**: Prevents numerical issues from large values

✅ **Transfer Learning**: Statistics carry over when loading pre-trained models

## Common Mistake in Your Code (FIXED!)

### Before Fix:
```python
# ob_rms was never updated - stayed at default (mean=0, std=1)
obs_normalized = (obs_batch - 0) / 1 = obs_batch  # No normalization!
```

### After Fix:
```python
# Update statistics before training
policy.ob_rms.update(all_training_obs)  # Learn mean & std

# Now normalization actually does something
obs_normalized = (obs_batch - learned_mean) / learned_std
```

## Visual Example

### Before Normalization:
```
Feature 1: [0.1, 0.2, 0.15, 0.18, ...]     (small)
Feature 2: [10.5, 12.3, 9.8, 11.2, ...]    (LARGE - dominates!)
Feature 3: [0.5, 0.6, 0.55, 0.52, ...]     (small)
```

### After Normalization:
```
Feature 1: [-0.5, 0.2, -0.1, 0.0, ...]     (equal importance)
Feature 2: [-0.3, 0.8, -0.5, 0.2, ...]     (equal importance)
Feature 3: [-0.2, 0.5, 0.1, -0.1, ...]     (equal importance)
```

## When to Update ob_rms

1. **During SFT**: Update once before training with all training data
2. **During RL**: Update continuously as new observations are collected
3. **At Test Time**: Use frozen statistics from training (don't update)

## Important Notes

⚠️ **Don't forget to save/load ob_rms**: It's part of `policy.state_dict()`

⚠️ **Clipping is important**: Outliers can still cause issues, so clip to [-5, 5]

⚠️ **Consistent normalization**: Always use the same statistics at train and test time
