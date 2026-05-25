#!/bin/sh
env="mujoco"
scenario="ManyAgentGoToGoalEnv-v0"
agent_conf="8x1"
agent_obsk=0
faulty_node=-1
#eval_faulty_node="-1 0 1 2 3 4 5"
eval_faulty_node=-1
algo="mat_dec"
# algo="mappo_dgnn_dsgd"
exp="single"
seed=00000

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
python train/eval_manygotogoal.py \
 --seed ${seed} \
 --truelyDistributedGNN True \
 --truelyDistributed True \
 --env_name ${env} \
 --algorithm_name ${algo} \
 --experiment_name ${exp} \
 --scenario ${scenario} \
 --agent_conf ${agent_conf} \
 --agent_obsk ${agent_obsk} \
 --faulty_node ${faulty_node} \
 --eval_faulty_node ${eval_faulty_node} \
 --model_dir "/home/i2r/shahil_ws/AERO-MARL/mat/scripts/results/mujoco/ManyAgentGoToGoalEnv-v0/mat_dec/single/wandb/run-20260211_102222-k4m0rdqe/files/transformer_1300.pt" \
 --allow_partial_restore \
 --clone_extra_agents_from 0 \
 --iterations 3 \
 --consensusLoss True \
 --gnn_loss_coef 10 \
 --critic_lr 2.5e-04 \
 --lr 2.5e-04 \
 --n_embd 128 \
 --value_loss_coef 1 \
 --max_grad_norm 0.6 \
 --eval_episodes 1 \
 --n_training_threads 32 \
 --n_rollout_threads 1 \
 --n_eval_rollout_threads 1 \
 --num_mini_batch 1 \
 --mini_batch_size 4000 \
 --episode_length 100 \
 --eval_interval 25 \
 --num_env_steps 200000000 \
 --ppo_epoch 10 \
 --gamma 0.95 \
 --gae_lambda 0.8 \
 --entropy_coef 0.001 \
 --clip_param 0.2 \
 --add_center_xy \
 --use_state_agent \
 --use_eval True\
 --n_quants 1 \
 --num-heads 1 \
 --num-layers 3 \
 --use_wandb True \
 --wandb_name "xxx" \
 --user_name "shahil-shaik7-clemson-university" \
 --use_render True \

# /home/i2r/shahil_ws/AERO-MARL/mat/scripts/results/mujoco/ManyAgentGoToGoalEnv-v0/mappo_dgnn_dsgd/single/wandb/run-20260210_210428-xebhrgyp/files/transformer_4900.pt
# /home/i2r/shahil_ws/AERO-MARL/mat/scripts/results/mujoco/ManyAgentGoToGoalEnv-v0/mat_dec/single/wandb/run-20260211_102222-k4m0rdqe/files/transformer_1300.pt