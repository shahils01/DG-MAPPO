#!/bin/bash

#SBATCH --job-name=mappo-dgnn-ap
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100gb
#SBATCH --time=72:00:00
#SBATCH --gpus-per-node=v100s:2

set -euo pipefail

# module load anaconda3
# source activate deepseek_env

cd "${SLURM_SUBMIT_DIR}"

# Single-process agent model parallelism: do not use torchrun.
srun python mat/scripts/train/train_smac.py \
  --truelyDistributed True \
  --consensusLoss True \
  --gnn_loss_coef 1 \
  --num-layers 3 \
  --iterations 5 \
  --env_name StarCraft2 \
  --algorithm_name mappo_dgnn_dsgd \
  --experiment_name mappo_dgnn_dsgd_agent_parallel \
  --map_name 10m_vs_11m \
  --eval_map_name 10m_vs_11m \
  --unit_sight_range 2 \
  --seed 0 \
  --n_training_threads 16 \
  --n_rollout_threads 32 \
  --episode_length 100 \
  --num_env_steps 40000000 \
  --buffer_device cpu \
  --mini_batch_size 128 \
  --ppo_epoch 5 \
  --lr 5e-4 \
  --gamma 0.98 \
  --gae_lambda 0.95 \
  --clip_param 0.05 \
  --entropy_coef 0.001 \
  --max_grad_norm 10 \
  --use_value_active_masks \
  --n_quants 1 \
  --hidden_size 64 \
  --hid-dim 64 \
  --n_embd 64 \
  --out_channels 64 \
  --num-heads 1 \
  --detach True \
  --agent_parallel \
  --agent_parallel_devices 0,1 \
  --use_eval
