#!/usr/bin/env bash
set -euo pipefail

env="smacv2"
# The bundled `terran` config resolves to the terran_5_vs_5 scenario. The
# scenario name itself is not a valid --smacv2_config alias in this repository.
map="terran"
algo="mat_dec"
exp="mat_dec"
seed=0

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
python train/train_smac.py   \
 --env_name ${env}   \
 --smacv2_config ${map}   \
 --algorithm_name ${algo}   \
 --experiment_name ${exp}   \
 --seed ${seed}   \
 --iterations 5   \
 --consensusLoss True   \
 --episode_length 200   \
 --num_env_steps 40000000   \
 --lr 5e-4   \
 --ppo_epoch 10   \
 --gamma 0.98   \
 --gae_lambda 0.95   \
 --clip_param 0.2   \
 --save_interval 100000   \
 --entropy_coef 0.01   \
 --max_grad_norm 10   \
 --use_eval   \
 --use_wandb True   \
 --wandb_name "xxx"   \
 --user_name "shahil-shaik7-clemson-university"
