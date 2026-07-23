#!/usr/bin/env bash
set -euo pipefail

env="StarCraft2"    # StarCraft2 or smacv2
map="10m_vs_11m"    # 6h_vs_8z, 5m_vs_6m, MMM2, protoss_5_vs_5
algo="mappo_dgnn_dsgd"       # Algos: {mappo_dgnn_dsgd, mat, mat_dec, ippo, consensus_ippo}
exp="mappo_dgnn_dsgd"
seed=0
hidden_dim=64
unit_sight_range=2

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_LAUNCH_BLOCKING=1 python train/train_smac.py \
 --truelyDistributed True \
 --num-layers 2 \
 --iterations 5 \
 --env_name ${env} \
 --algorithm_name ${algo} \
 --experiment_name ${exp} \
 --map_name ${map} \
 --eval_map_name ${map} \
 --unit_sight_range ${unit_sight_range} \
 --seed ${seed} \
 --n_training_threads 32 \
 --n_rollout_threads 32 \
 --mini_batch_size 1600 \
 --episode_length 200 \
 --num_env_steps 40000000 \
 --lr 5e-4 \
 --ppo_epoch 10 \
 --gamma 0.995 \
 --gae_lambda 0.99 \
 --clip_param 0.1 \
 --save_interval 100000 \
 --use_value_active_masks \
 --entropy_coef 0.01 \
 --max_grad_norm 10 \
 --n_quants 1 \
 --hidden_size ${hidden_dim} \
 --hid-dim ${hidden_dim} \
 --n_embd ${hidden_dim} \
 --out_channels ${hidden_dim} \
 --num-heads 1 \
 --detach True \
 --share_policy \
 --use_wandb True \
 --wandb_name "xxx" \
 --user_name "shahil-shaik7-clemson-university"

#  --use_eval \
# WANDB_MODE=offline
# If smac fails, enter the command: pkill -f "SC2_x64 -listen"
# --truelyDistributed True
# --gpu-freq=high,memory=high
# --detach True
