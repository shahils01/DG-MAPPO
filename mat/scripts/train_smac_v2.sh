#!/usr/bin/env bash
set -euo pipefail

env="smacv2"    # StarCraft2 or smacv2
map="terran_epo"    # 6h_vs_8z, 5m_vs_6m, MMM2, protoss_5_vs_5
algo="mappo_dgnn_dsgd"       # Algos: {mappo_dgnn_dsgd, mat, mat_dec, ippo, consensus_ippo}
exp="mappo_dgnn_dsgd"
seed=0
hidden_dim=64
unit_sight_range=2

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
python train/train_smac.py   \
 --env_name ${env}   \
 --smacv2_config ${map}   \
 --algorithm_name ${algo}   \
 --experiment_name ${exp}   \
 --iterations 5   \
 --encode_state True   \
 --consensusLoss True   \
 --episode_length 200   \
 --num_env_steps 40000000   \
 --lr 5e-4   \
 --ppo_epoch 10   \
 --gamma 0.99   \
 --gae_lambda 0.95   \
 --clip_param 0.05   \
 --save_interval 100000   \
 --entropy_coef 0.01   \
 --max_grad_norm 10   \
 --use_eval   \
 --use_wandb True   \
 --wandb_name "xxx"   \
 --user_name "shahil-shaik7-clemson-university"