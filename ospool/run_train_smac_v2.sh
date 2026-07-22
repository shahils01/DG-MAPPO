#!/usr/bin/env bash
set -euo pipefail

echo "Host: $(hostname)"
echo "Working directory: $PWD"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

# HTCondor is configured to transfer this directory on every exit. Create it
# before validation so an early failure reports its real exit code instead of
# being masked by a missing-output-directory transfer error.
rm -rf mat/scripts/results
mkdir -p mat/scripts/results

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "ERROR: nvidia-smi is unavailable inside the job." >&2
  exit 10
fi

# Override this in the container environment if SC2 is installed elsewhere.
export SC2PATH="${SC2PATH:-/opt/StarCraftII}"
required_map="${SC2PATH}/Maps/SMAC_Maps/32x32_flat.SC2Map"

if ! find "${SC2PATH}/Versions" -maxdepth 2 -type f -name SC2_x64 \
  -perm -u+x -print -quit 2>/dev/null | grep -q .; then
  echo "ERROR: StarCraft II was not found below SC2PATH=${SC2PATH}." >&2
  exit 12
fi

if [[ ! -f "${required_map}" ]]; then
  echo "ERROR: Required SMACv2 map is missing: ${required_map}" >&2
  exit 13
fi

python - <<'PY'
import torch
import smacv2

assert torch.cuda.is_available(), "PyTorch cannot access CUDA"
print(f"PyTorch: {torch.__version__}")
print(f"CUDA runtime: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"SMACv2: {smacv2.__file__}")
PY

# The submit file transfers ospool/.wandb_api_key. Depending on the HTCondor
# file-transfer layout, a single transferred file may arrive at the sandbox
# root or retain its relative path, so accept either location.
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  for wandb_key_file in .wandb_api_key ospool/.wandb_api_key; do
    if [[ -s "${wandb_key_file}" ]]; then
      WANDB_API_KEY="$(tr -d '\r\n' < "${wandb_key_file}")"
      export WANDB_API_KEY
      break
    fi
  done
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "ERROR: the transferred W&B API key is missing or empty." >&2
  exit 14
fi

export WANDB_MODE=online
echo "W&B online logging enabled."

cd mat/scripts
exec bash train_smac_v2.sh
