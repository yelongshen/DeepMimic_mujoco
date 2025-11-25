# TensorFlow 2.x Compatibility

The DeepMimic codebase was originally written for **TensorFlow 1.x**, which has significant API differences from **TensorFlow 2.x**.

## Issue

When running with TensorFlow 2.x, you'll encounter errors like:

```python
AttributeError: module 'tensorflow' has no attribute 'ConfigProto'
AttributeError: module 'tensorflow' has no attribute 'Session'
```

## Solution

We've added **TensorFlow 2.x compatibility mode** to the following files:

- `src/utils/tf_util.py`
- `src/trpo.py`
- `src/mlp_policy_trpo.py`

### What was changed:

```python
# Added at the top of each file after tensorflow import:
import tensorflow as tf

# TensorFlow 2.x compatibility
if hasattr(tf, '__version__') and int(tf.__version__.split('.')[0]) >= 2:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
```

This enables **TensorFlow 1.x compatibility mode** when TensorFlow 2.x is detected.

## Installation Options

### Option 1: Use TensorFlow 2.x with compatibility mode (Recommended)

```bash
pip install tensorflow>=2.0
```

The code will automatically use `tensorflow.compat.v1` API.

**Pros:**
- ✅ Modern TensorFlow version
- ✅ Better Python 3.8+ support
- ✅ GPU support with recent CUDA versions
- ✅ Still maintained by Google

**Cons:**
- ⚠️ Uses deprecated TF1 API (but still works)

### Option 2: Use TensorFlow 1.x (Legacy)

```bash
pip install tensorflow==1.15.0  # Last 1.x version
```

**Pros:**
- ✅ Native TF1 API support
- ✅ No compatibility layer needed

**Cons:**
- ❌ Only supports Python 3.7 or older
- ❌ No longer maintained
- ❌ Requires old CUDA versions (CUDA 10.0)
- ❌ Security vulnerabilities

## Recommended Setup

For **Python 3.8+** (including Python 3.12):

```bash
# Install TensorFlow 2.x
pip install tensorflow>=2.10

# Or for GPU support:
pip install tensorflow[and-cuda]>=2.10
```

The compatibility layer will automatically activate.

## Verification

Check that it works:

```bash
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
```

Then run:

```bash
cd src
python trpo.py --task evaluate --load_model_path checkpoint_tmp/DeepMimic/trpo-walk-0/DeepMimic/trpo-walk-0
```

You should see:

```
Using TensorFlow 2.x.x with v1 compatibility mode
```

## Known Limitations

1. **TF2 eager execution is disabled** - The code runs in graph mode like TF1
2. **Some TF2 features unavailable** - Can't use tf.keras, tf.data in the modern way
3. **Deprecation warnings** - You may see warnings about deprecated TF1 APIs

These are expected and don't affect functionality.

## Future Work

For a complete TensorFlow 2.x migration (without compatibility mode):

1. Convert `tf.Session` → `tf.function` decorators
2. Replace `placeholders` → `tf.Variable` or function arguments
3. Update optimizer APIs
4. Rewrite custom training loops

This would require significant code refactoring but would enable full TF2 features.

## Troubleshooting

### "No module named 'tensorflow.compat.v1'"

Your TensorFlow version is too old. Upgrade:

```bash
pip install --upgrade tensorflow>=2.0
```

### GPU not detected

For TensorFlow 2.10+:

```bash
pip install tensorflow[and-cuda]>=2.10
```

This installs CUDA dependencies automatically.

### Still getting TF1 API errors

Make sure the compatibility code is at the **top of the file** before any TF operations:

```python
import tensorflow as tf
if int(tf.__version__.split('.')[0]) >= 2:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
# Now use tf as normal
```

## Summary

✅ **TensorFlow 2.x is now supported** through compatibility mode  
✅ **Works with Python 3.8-3.12**  
✅ **No code changes needed for users**  
⚠️ Uses deprecated TF1 APIs (but still functional)
