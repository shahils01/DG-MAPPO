#!/bin/bash
# 4 algorithms x 3 larger-team conditions x 5 seeds = 60 tasks.
# Conditions: 8v4 easy (prey at 0.85x speed), 8v4 hard and 8v6 hard
# (prey at equal speed). Submit from the repository root with:
#   mkdir -p /scratch/shahils/DG-MAPPO-results/{slurm,predator_prey_large}
#   sbatch mat/scripts/palmetto/run_predator_prey_large_team_array.sh

#SBATCH --job-name=pp-large
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --gpus-per-node=1
#SBATCH --array=0-59%8
#SBATCH --output=/scratch/shahils/DG-MAPPO-results/slurm/%x_%A_%a.out
#SBATCH --error=/scratch/shahils/DG-MAPPO-results/slurm/%x_%A_%a.err

REPO_ROOT="${REPO_ROOT:-$HOME/Desktop/gitBackupRepo/DG-MAPPO}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/$USER/DG-MAPPO-results/predator_prey_large}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must be submitted as a Slurm array}"

algorithms=(mappo_dgnn_dsgd mat_dec ippo consensus_ippo)
conditions=(8v4_easy 8v4_hard 8v6_hard)
predators=(8 8 8)
prey=(4 4 6)
prey_speed_ratios=(0.85 1.0 1.0)
seeds=(0 1 2 3 4)

algorithm_index=$((TASK_ID / 15))
condition_index=$(((TASK_ID % 15) / 5))
seed_index=$((TASK_ID % 5))
algorithm="${algorithms[$algorithm_index]}"
condition="${conditions[$condition_index]}"
num_predators="${predators[$condition_index]}"
num_prey="${prey[$condition_index]}"
prey_speed_ratio="${prey_speed_ratios[$condition_index]}"
seed="${seeds[$seed_index]}"
experiment_name="large_team_${condition}"
run_dir="${OUTPUT_ROOT}/${condition}/${algorithm}/seed${seed}"

# Slurm starts a non-interactive shell; initialize Palmetto modules first.
source /etc/profile.d/modules.sh
module load anaconda3
source activate deepseek_env
set -euo pipefail

cd "${REPO_ROOT}/mat/scripts"
mkdir -p "${run_dir}"

gru_args=()
if [[ "${algorithm}" != "mat_dec" ]]; then
  gru_args=(--use_actor_gru --use_critic_gru --recurrent_N 1 --data_chunk_length 20)
fi

# DG-MAPPO recomputes graph messages during PPO updates.  With 32 rollout
# environments, four recurrent minibatches place 80 chunks on one GPU and
# exceed 44 GB on the available accelerators.  Sixteen keeps the graph batch
# at 20 chunks while leaving the other baselines unchanged.
num_mini_batch=4
if [[ "${algorithm}" == "mappo_dgnn_dsgd" ]]; then
  num_mini_batch=16
fi

echo "algorithm=${algorithm} condition=${condition} seed=${seed} run_dir=${run_dir}"
python train/train_long_range_predator_prey.py \
  --seed "${seed}" \
  --truelyDistributed True \
  --env_name long_range_predator_prey \
  --algorithm_name "${algorithm}" \
  --experiment_name "${experiment_name}" \
  --scenario LongRangePredatorPreyContinuous-v0 \
  --run_dir "${run_dir}" \
  --num_predators "${num_predators}" \
  --predator_max_speed 0.7 \
  --num_prey "${num_prey}" \
  --world_size 6.0 \
  --random_start_positions \
  --init_min_predator_dist 0.45 \
  --init_min_prey_dist 0.45 \
  --init_min_prey_predator_dist 1.0 \
  --obs_radius 1.8 \
  --comm_radius 2.2 \
  --capture_radius 0.35 \
  --capture_k 1 \
  --prey_speed_ratio "${prey_speed_ratio}" \
  --collision_radius 0.18 \
  --env_device cpu \
  --faulty_node -1 \
  --eval_faulty_node -1 \
  --iterations 3 \
  --gnn_loss_coef 1 \
  --critic_lr 5e-4 \
  --lr 5e-4 \
  --n_embd 128 \
  --hidden_size 128 \
  --out_channels 128 \
  --value_loss_coef 1 \
  --max_grad_norm 0.8 \
  --eval_episodes 5 \
  --n_training_threads 16 \
  --n_rollout_threads 32 \
  --n_eval_rollout_threads 1 \
  --num_mini_batch "${num_mini_batch}" \
  --mini_batch_size 1600 \
  --episode_length 200 \
  --env_episode_length 200 \
  --eval_interval 25 \
  --num_env_steps 20000000 \
  --ppo_epoch 10 \
  --gamma 0.99 \
  --gae_lambda 0.95 \
  --entropy_coef 0.01 \
  --clip_param 0.2 \
  --add_center_xy \
  --use_state_agent \
  --use_eval \
  --n_quants 1 \
  --num-heads 1 \
  --num-layers 3 \
  --encode_state True \
  --use_wandb True \
  --user_name shahil-shaik7-clemson-university \
  "${gru_args[@]}"
