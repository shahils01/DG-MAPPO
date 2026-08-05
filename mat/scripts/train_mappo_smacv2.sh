#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Native MAPPO baseline: shared local actor + centralized state critic.
env="smacv2"
map="protoss_epo"
seed=0
exp="mappo_epo"

echo "env=${env}, map=${map}, algorithm=mappo, experiment=${exp}, seed=${seed}"
python train/train_smac.py \
  --env_name "${env}" \
  --smacv2_config "${map}" \
  --algorithm_name mappo \
  --experiment_name "${exp}" \
  --seed "${seed}" \
  --n_embd 64 \
  --n_training_threads 16 \
  --n_rollout_threads 32 \
  --episode_length 200 \
  --mini_batch_size 1600 \
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
  --user_name "shahil-shaik7-clemson-university"
