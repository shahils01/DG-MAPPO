#!/bin/bash
# Five seeds for each of {IPPO-GRU, consensus-IPPO-GRU} on the three EPO races.
# Submit from the repository root with:
#   mkdir -p /scratch/shahils/DG-MAPPO-results/{slurm,smacv2_ippo_gru}
#   sbatch mat/scripts/palmetto/run_ippo_gru_smacv2_array.sh

#SBATCH --job-name=ippo-gru-epo
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=72:00:00
#SBATCH --gpus-per-node=1
#SBATCH --array=0-29%4
#SBATCH --output=/scratch/shahils/DG-MAPPO-results/slurm/%x_%A_%a.out
#SBATCH --error=/scratch/shahils/DG-MAPPO-results/slurm/%x_%A_%a.err

REPO_ROOT="${REPO_ROOT:-$HOME/Desktop/gitBackupRepo/DG-MAPPO}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/$USER/DG-MAPPO-results/smacv2_ippo_gru}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must be submitted as a Slurm array}"

algorithms=(ippo consensus_ippo)
maps=(terran_epo protoss_epo zerg_epo)
seeds=(0 1 2 3 4)

algorithm_index=$((TASK_ID / 15))
map_index=$(((TASK_ID % 15) / 5))
seed_index=$((TASK_ID % 5))
algorithm="${algorithms[$algorithm_index]}"
map="${maps[$map_index]}"
seed="${seeds[$seed_index]}"
experiment_name="${algorithm}_gru_epo"
run_dir="${OUTPUT_ROOT}/${map}/${algorithm}/seed${seed}"

# Slurm invokes a non-interactive shell, so Palmetto's module function is not
# initialized unless we source it explicitly.
source /etc/profile.d/modules.sh
module load anaconda3
source activate deepseek_env

set -euo pipefail

cd "${REPO_ROOT}/mat/scripts"
mkdir -p "${run_dir}"

echo "algorithm=${algorithm} map=${map} seed=${seed} run_dir=${run_dir}"
python train/train_smac.py \
  --env_name smacv2 \
  --smacv2_config "${map}" \
  --algorithm_name "${algorithm}" \
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
