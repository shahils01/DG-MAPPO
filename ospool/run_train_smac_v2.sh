#!/usr/bin/env bash
set -euo pipefail

echo "Host: $(hostname)"
echo "Working directory: $PWD"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "ERROR: nvidia-smi is unavailable inside the job." >&2
  exit 10
fi

gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
if [[ "${gpu_name}" != *A100* ]]; then
  echo "ERROR: HTCondor assigned '${gpu_name}', not an NVIDIA A100." >&2
  exit 11
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

# The launcher enables W&B. Avoid aborting an otherwise valid batch job when
# no credential was provisioned in the container/job environment.
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  export WANDB_MODE=offline
  echo "WANDB_API_KEY is unset; recording the W&B run offline."
fi

# Do not send old local results back as output from this job sandbox.
rm -rf mat/scripts/results
mkdir -p mat/scripts/results

cd mat/scripts
exec bash train_smac_v2.sh
