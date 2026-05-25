#!/bin/sh
env="StarCraft2"    # StarCraft2 or smacv2
map="5m_vs_6m"    # 6h_vs_8z, 5m_vs_6m, MMM2, protoss_5_vs_5
algo="mappo_dgnn_dsgd"
exp="single"
seed=0
hidden_dim=128

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_LAUNCH_BLOCKING=1 python train/train_smac.py \
 --truelyDistributedGNN True \
 --truelyDistributed True \
 --gnn_loss_coef 20 \
 --lambd-gnn 0 \
 --num-layers 3 \
 --iterations 5 \
 --sight_range 1 \
 --meanGNN True \
 --env_name ${env} \
 --algorithm_name ${algo} \
 --experiment_name ${exp} \
 --map_name ${map} \
 --eval_map_name ${map} \
 --seed ${seed} \
 --n_training_threads 32 \
 --n_rollout_threads 32 \
 --num_mini_batch 1 \
 --mini_batch_size 3200 \
 --episode_length 500 \
 --num_env_steps 40000000 \
 --lr 5e-4 \
 --ppo_epoch 5 \
 --gamma 0.98 \
 --gae_lambda 0.95 \
 --clip_param 0.05 \
 --save_interval 100000 \
 --use_value_active_masks \
 --entropy_coef 0.001 \
 --max_grad_norm 10 \
 --encode_state True \
 --n_quants 1 \
 --hidden_size ${hidden_dim} \
 --hid-dim ${hidden_dim} \
 --n_embd ${hidden_dim} \
 --out_channels ${hidden_dim} \
 --num-heads 1 \
 --detach True \
 --use_eval \
 --use_wandb True \
 --wandb_name "xxx" \
 --user_name "shahil-shaik7-clemson-university"

# WANDB_MODE=offline
# If smac fails, enter the command: pkill -f "SC2_x64 -listen"
# --truelyDistributed True
# --gpu-freq=high,memory=high
# --detach True
