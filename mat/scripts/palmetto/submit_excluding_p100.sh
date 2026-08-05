#!/bin/bash
# Submit a Slurm script while excluding all P100 nodes currently in work1.
# Usage: submit_excluding_p100.sh [sbatch options] <script> [script arguments]

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 [sbatch options] <script> [script arguments]" >&2
  exit 2
fi

p100_nodes=$(sinfo -N -h -p work1 -o "%N %G" \
  | awk '$2 ~ /gpu:p100/ {print $1}' \
  | sort -u \
  | paste -sd, -)

if [[ -z "${p100_nodes}" ]]; then
  echo "Could not resolve P100 nodes; refusing to submit without an exclusion list." >&2
  exit 1
fi

exec sbatch --exclude="${p100_nodes}" "$@"
