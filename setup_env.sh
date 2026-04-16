#!/bin/bash
# Environment setup for ContinuousBenchEval.
# Creates a conda env (default name: cbe) and installs the picked backend.
#
# Usage:
#   bash setup_env.sh torch-gpu                     # cbe + HF/TRL on GPU
#   bash setup_env.sh jax-gpu                       # cbe + Kauldron on GPU
#   bash setup_env.sh jax-tpu                       # cbe + Kauldron on TPU
#
# Add wandb:
#   bash setup_env.sh torch-gpu wandb
#
# Custom env name (third positional arg):
#   bash setup_env.sh jax-gpu "" cbe-debug

set -e

BACKEND="${1:?Usage: bash setup_env.sh <torch-gpu|jax-tpu|jax-gpu> [wandb] [env_name]}"
EXTRA="${2:-}"
ENV_NAME="${3:-cbe}"

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

# JAX-GPU: pip-installed nvidia-*-cu12 libs aren't on LD_LIBRARY_PATH by default,
# so JAX can't dlopen libcusparse / libcublas / etc. and falls back to CPU.
# Register an activation hook that prepends those dirs to LD_LIBRARY_PATH
# every time the env is activated.
if [[ "${BACKEND}" == "jax-gpu" ]]; then
    echo "=== Registering activation hook for NVIDIA lib paths ==="
    ENV_PREFIX="$(conda info --base)/envs/${ENV_NAME}"
    ACTIVATE_DIR="${ENV_PREFIX}/etc/conda/activate.d"
    DEACTIVATE_DIR="${ENV_PREFIX}/etc/conda/deactivate.d"
    mkdir -p "${ACTIVATE_DIR}" "${DEACTIVATE_DIR}"

    cat > "${ACTIVATE_DIR}/cbe_nvidia_libs.sh" <<'ACTIVATE_EOF'
# Prepend pip-installed NVIDIA CUDA lib dirs so JAX can dlopen them.
_CBE_NVIDIA_BASE="${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia"
if [ -d "${_CBE_NVIDIA_BASE}" ]; then
    _CBE_NVIDIA_LIBS=$(find "${_CBE_NVIDIA_BASE}" -name lib -type d | tr '\n' ':')
    export _CBE_LD_LIBRARY_PATH_SAVE="${LD_LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="${_CBE_NVIDIA_LIBS}${LD_LIBRARY_PATH:-}"
fi
unset _CBE_NVIDIA_BASE _CBE_NVIDIA_LIBS
ACTIVATE_EOF

    cat > "${DEACTIVATE_DIR}/cbe_nvidia_libs.sh" <<'DEACTIVATE_EOF'
# Restore LD_LIBRARY_PATH to what it was before the env was activated.
if [ -n "${_CBE_LD_LIBRARY_PATH_SAVE+x}" ]; then
    export LD_LIBRARY_PATH="${_CBE_LD_LIBRARY_PATH_SAVE}"
    unset _CBE_LD_LIBRARY_PATH_SAVE
fi
DEACTIVATE_EOF
fi

echo ""
echo "=== Done! Activate with: conda activate ${ENV_NAME} ==="
echo ""
echo "Quick start:"
echo "  python train.py --config configs/tracks/news.yaml --framework hf"
echo "  python train.py --config configs/tracks/news.yaml --framework kd"
