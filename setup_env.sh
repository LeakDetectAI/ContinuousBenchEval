#!/bin/bash
# Environment setup for ContinuousBenchEval
#
# Usage:
#   bash setup_env.sh torch-gpu     # HuggingFace/TRL on GPU
#   bash setup_env.sh jax-tpu       # Kauldron on TPU
#   bash setup_env.sh jax-gpu       # Kauldron on GPU
#
# Add wandb support:
#   bash setup_env.sh torch-gpu wandb

set -e

BACKEND="${1:?Usage: bash setup_env.sh <torch-gpu|jax-tpu|jax-gpu> [wandb]}"
EXTRA="${2:-}"

ENV_NAME="cbe-${BACKEND}"

echo "=== Creating conda environment: ${ENV_NAME} ==="
conda create -n "${ENV_NAME}" python=3.11 -y
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "=== Installing pip dependencies ==="
pip install -U pip setuptools wheel

# Install framework-specific requirements
pip install -r "requirements/${BACKEND}.txt"

# Install the cbe package in editable mode
pip install -e .

# Optional: wandb
if [ "${EXTRA}" = "wandb" ]; then
    echo "=== Installing wandb ==="
    pip install -r requirements/wandb.txt
fi

# JAX-specific: ensure libstdc++ is up to date and TF is CPU-only
if [[ "${BACKEND}" == jax-* ]]; then
    echo "=== JAX post-install: updating libstdc++ and ensuring tensorflow-cpu ==="
    conda install -y -c conda-forge "libstdcxx-ng>=14" "libgcc-ng>=14" icu
    pip uninstall -y tensorflow 2>/dev/null || true
    pip install -U tensorflow-cpu
fi

echo ""
echo "=== Done! Activate with: conda activate ${ENV_NAME} ==="
echo ""
echo "Quick start:"
echo "  python train.py --config configs/tracks/news.yaml --framework hf"
echo "  python train.py --config configs/tracks/news.yaml --framework kd"
