#!/bin/bash
# Five-seed recurrent MAPPO baseline on each SMACv2 EPO race.
# Submit from the repository root with:
#   mkdir -p /scratch/shahils/DG-MAPPO-results/{slurm,smacv2_mappo_gru}
#   sbatch mat/scripts/palmetto/run_mappo_gru_smacv2_array.sh

#SBATCH --job-name=mappo-gru-epo
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=1
#SBATCH --array=0-14%4
#SBATCH --output=/scratch/shahils/DG-MAPPO-results/slurm/%x_%A_%a.out
#SBATCH --error=/scratch/shahils/DG-MAPPO-results/slurm/%x_%A_%a.err

set -eo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/Desktop/gitBackupRepo/DG-MAPPO}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/$USER/DG-MAPPO-results/smacv2_mappo_gru}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must be submitted as a Slurm array}"

maps=(terran_epo protoss_epo zerg_epo)
seeds=(0 1 2 3 4)

map_index=$((TASK_ID / 5))
seed_index=$((TASK_ID % 5))
map="${maps[$map_index]}"
seed="${seeds[$seed_index]}"
experiment_name="mappo_gru_epo"
run_dir="${OUTPUT_ROOT}/${map}/mappo/seed${seed}"

# Slurm starts a non-interactive shell; initialize Palmetto modules first.
source /etc/profile.d/modules.sh
module load anaconda3
source activate deepseek_env
set -u

cd "${REPO_ROOT}/mat/scripts"
mkdir -p "${run_dir}"

echo "algorithm=mappo map=${map} seed=${seed} run_dir=${run_dir}"
python train/train_smac.py \
  --env_name smacv2 \
  --smacv2_config "${map}" \
  --algorithm_name mappo \
  --experiment_name "${experiment_name}" \
  --seed "${seed}" \
  --run_dir "${run_dir}" \
  --use_actor_gru \
  --use_critic_gru \
  --recurrent_N 1 \
  --data_chunk_length 20 \
  --n_embd 64 \
  --n_training_threads 16 \
  --n_rollout_threads 8 \
  --episode_length 200 \
  --mini_batch_size 800 \
  --num_env_steps 20000000 \
  --lr 5e-4 \
  --ppo_epoch 10 \
  --gamma 0.99 \
  --gae_lambda 0.95 \
  --clip_param 0.1 \
  --save_interval 100000 \
  --entropy_coef 0.01 \
  --max_grad_norm 10 \
  --use_eval \
  --use_wandb True \
  --user_name shahil-shaik7-clemson-university
