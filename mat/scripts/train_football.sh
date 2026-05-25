#!/bin/sh
env="football"
scenario="academy_counterattack_easy"
# academy_pass_and_shoot_with_keeper
# academy_3_vs_1_with_keeper
# academy_counterattack_easy
n_agent=11
algo="mappo_dgnn_dsgd"
exp="single"
seed=0

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=0 apptainer exec --nv /home/shahils/gfootball_apptainer/gfootball.sif \
 python3 train/train_football.py \
 --seed ${seed} \
 --env_name ${env} \
 --algorithm_name ${algo} \
 --experiment_name ${exp} \
 --scenario ${scenario} \
 --n_agent ${n_agent} \
 --iterations 6 \
 --truelyDistributed True \
 --truelyDistributedGNN True \
 --lr 5e-4 \
 --entropy_coef 0.001 \
 --max_grad_norm 0.5 \
 --eval_episodes 32 \
 --n_training_threads 32 \
 --n_rollout_threads 32 \
 --mini_batch_size 4000 \
 --num_mini_batch 1 \
 --episode_length 200 \
 --eval_interval 25 \
 --num_env_steps 10000000 \
 --ppo_epoch 10 \
 --clip_param 0.2 \
 --use_eval \
 --use_value_active_masks \
 --use_policy_active_masks \
 --use_wandb True \
 --wandb_name "xxx" \
 --user_name "shahil-shaik7-clemson-university"
