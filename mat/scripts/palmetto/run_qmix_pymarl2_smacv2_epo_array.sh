#!/bin/bash
# Canonical PyMARL2 QMIX baseline for the three SMACv2 6-vs-5 EPO races.
#
# Full paper run:
#   sbatch mat/scripts/palmetto/run_qmix_pymarl2_smacv2_epo_array.sh
# Short smoke test (Terran, seed 0):
#   sbatch --array=0 --time=01:00:00 --export=ALL,T_MAX=100000,SAVE_MODEL=False \
#     mat/scripts/palmetto/run_qmix_pymarl2_smacv2_epo_array.sh

#SBATCH --job-name=qmix-epo
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=1
#SBATCH --array=0-14%4
#SBATCH --output=/scratch/shahils/DG-MAPPO-results/slurm/%x_%A_%a.out
#SBATCH --error=/scratch/shahils/DG-MAPPO-results/slurm/%x_%A_%a.err

set -eo pipefail

PYMARL_ROOT="${PYMARL_ROOT:-$HOME/Desktop/gitBackupRepo/pymarl2-smacv2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/$USER/DG-MAPPO-results/smacv2_qmix_pymarl2}"
SC2PATH="${SC2PATH:-$HOME/StarCraftII}"
T_MAX="${T_MAX:-20000000}"
SAVE_MODEL="${SAVE_MODEL:-True}"
USE_CUDA="${USE_CUDA:-True}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_GROUP="${WANDB_GROUP:-qmix_epo}"
RUN_LABEL_PREFIX="${RUN_LABEL_PREFIX:-qmix_epo}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must be submitted as a Slurm array}"

env_configs=(sc2_gen_terran_epo sc2_gen_protoss_epo sc2_gen_zerg_epo)
scenario_names=(terran_6_vs_5_epo protoss_6_vs_5_epo zerg_6_vs_5_epo)
seeds=(0 1 2 3 4)

map_index=$((TASK_ID / 5))
seed_index=$((TASK_ID % 5))
env_config="${env_configs[$map_index]}"
scenario_name="${scenario_names[$map_index]}"
seed="${seeds[$seed_index]}"
project="${scenario_name}_journal_new"
run_dir="${OUTPUT_ROOT}/${scenario_name}/qmix/seed${seed}"

source /etc/profile.d/modules.sh
module load anaconda3
source activate pymarl2_epo
set -u

export SC2PATH
export WANDB_MODE
export WANDB_SILENT=true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "${PYMARL_ROOT}"
mkdir -p "${run_dir}"

echo "algorithm=qmix scenario=${scenario_name} seed=${seed} t_max=${T_MAX} run_dir=${run_dir}"
python src/main.py \
  --config=qmix \
  --env-config="${env_config}" \
  with \
  seed="${seed}" \
  env_args.capability_config.n_units=6 \
  env_args.capability_config.n_enemies=5 \
  env_args.prob_obs_enemy=0.5 \
  env_args.action_mask=False \
  t_max="${T_MAX}" \
  gamma=0.99 \
  td_lambda=0.4 \
  epsilon_anneal_time=100000 \
  test_interval=10000 \
  test_nepisode=32 \
  use_cuda="${USE_CUDA}" \
  use_wandb=True \
  project="${project}" \
  entity=shahil-shaik7-clemson-university \
  group="${WANDB_GROUP}" \
  label="${RUN_LABEL_PREFIX}_seed${seed}" \
  save_model="${SAVE_MODEL}" \
  save_model_interval=2000000 \
  local_results_path="${run_dir}"
