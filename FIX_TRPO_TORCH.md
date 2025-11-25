# Quick Fix for trpo_torch.py

The file got corrupted during editing. Here's how to fix it:

## Option 1: Restore from Git (EASIEST)

```bash
cd /mnt/c/DeepMimic_mujoco
git checkout src/trpo_torch.py
```

Then manually apply just the key fix below.

## Option 2: Manual Fix

The file has duplicate/corrupted content at the top. 

**The key fix needed** is in the `update_policy()` method around line 153-210.

Find this section:
```python
# Get current policy outputs
_, _, pd = self.policy(obs, stochastic=True)

# Compute surrogate loss
log_probs = -pd.neglogp(actions)
ratio = torch.exp(log_probs - old_log_probs)
surr_loss = -(ratio * advantages).mean()
```

**Replace with:**
```python
# IMPORTANT: Get OLD policy outputs and store them (before any updates)
with torch.no_grad():
    _, _, old_pd = self.policy(obs, stochastic=True)
    old_log_probs_actual = -old_pd.neglogp(actions)

# Compute surrogate loss with OLD policy (before update)
_, _, pd = self.policy(obs, stochastic=True)
log_probs = -pd.neglogp(actions)
ratio = torch.exp(log_probs - old_log_probs_actual)  # Use stored old log probs
surr_loss = -(ratio * advantages).mean()
```

Then in the line search section (around line 230):
```python
success, new_params = self._line_search(
    old_params, fullstep, expected_improve, 
    obs, actions, advantages, old_log_probs,  # ← Change this
    surr_loss,
    policy_params
)
```

**Change to:**
```python
success, new_params = self._line_search(
    old_params, fullstep, expected_improve, 
    obs, actions, advantages, old_log_probs_actual,  # ← Use actual old log probs
    surr_loss,
    policy_params
)
```

## What This Fixes

The ratio was always 1.0 because we were comparing the policy against itself BEFORE updating. Now we:
1. Store the old policy distribution
2. Update parameters  
3. Compare new vs old (should give ratio ≠ 1.0)

## Test After Fix

Run training and check:
```
Ratio: mean=1.002345, std=0.012345  ← Should NOT be exactly 1.0!
KL divergence: 0.005123  ← Should be > 0!
```

If you still see ratio=1.000000 exactly, the fix wasn't applied correctly.
