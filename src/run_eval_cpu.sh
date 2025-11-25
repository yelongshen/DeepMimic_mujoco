#!/bin/bash
# Force CPU-only execution to avoid CUDA/GPU issues
export CUDA_VISIBLE_DEVICES=""
export TF_CPP_MIN_LOG_LEVEL="2"
export TF_FORCE_GPU_ALLOW_GROWTH="false"
export TF_GPU_ALLOCATOR="cuda_malloc_async"

# Prevent TensorFlow from loading CUDA libraries
export LD_PRELOAD=""

# Run the evaluation
xvfb-run -a python trpo_torch.py --task evaluate --load_model_path policy_sft_pretrained.pth "$@"

#xvfb-run -a python trpo_torch.py --task evaluate --load_model_path checkpoint_torch/trpo-dance_a-0/iter_2400.pt "$@"
