#!/bin/sh

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "${SCRIPT_DIR}" || exit 1

env="long_range_predator_prey"
scenario="LongRangePredatorPreyContinuous-v0"

# Override these without editing:
#   ALGO=ippo NUM_PREDATORS=8 NUM_PREY=3 sh train_predator_prey.sh
algo="${ALGO:-mappo_dgnn_dsgd}"
exp="${EXP:-single}"
seed="${SEED:-1}"
model_dir="${MODEL_DIR:-}"

num_predators="${NUM_PREDATORS:-6}"
predator_max_speed="${predator_max_speed:-0.7}"
num_prey="${NUM_PREY:-2}"
world_size="${WORLD_SIZE:-6.0}"
obs_radius="${OBS_RADIUS:-1.8}"
comm_radius="${COMM_RADIUS:-2.2}"
ensure_connected_comm_graph="${ENSURE_CONNECTED_COMM_GRAPH:-True}"
ensure_prey_visible="${ENSURE_PREY_VISIBLE:-True}"
capture_radius="${CAPTURE_RADIUS:-0.35}"
capture_k="${CAPTURE_K:-1}"
prey_speed_ratio="${PREY_SPEED_RATIO:-0.85}"
collision_radius="${COLLISION_RADIUS:-0.18}"
env_device="${ENV_DEVICE:-cpu}"
n_training_threads="${N_TRAINING_THREADS:-32}"
n_rollout_threads="${N_ROLLOUT_THREADS:-32}"
n_eval_rollout_threads="${N_EVAL_ROLLOUT_THREADS:-1}"
episode_length="${EPISODE_LENGTH:-200}"
env_episode_length="${ENV_EPISODE_LENGTH:-${episode_length}}"

faulty_node="${FAULTY_NODE:--1}"
eval_faulty_node="${EVAL_FAULTY_NODE:--1}"
hidden_dim="${HIDDEN_DIM:-128}"
use_wandb="${USE_WANDB:-True}"
user_name="${USER_NAME:-shahil-shaik7-clemson-university}"

echo "env=${env}, scenario=${scenario}, algo=${algo}, exp=${exp}, seed=${seed}"
echo "num_predators=${num_predators}, num_prey=${num_prey}, comm_radius=${comm_radius}, obs_radius=${obs_radius}"
echo "rollout episode_length=${episode_length}, env_episode_length=${env_episode_length}"

graph_args=""
if [ "${ensure_connected_comm_graph}" = "False" ] || [ "${ensure_connected_comm_graph}" = "false" ] || [ "${ensure_connected_comm_graph}" = "0" ]; then
  graph_args="--disable_connected_comm_graph"
  echo "connected communication graph guarantee disabled"
fi

visibility_args=""
if [ "${ensure_prey_visible}" = "False" ] || [ "${ensure_prey_visible}" = "false" ] || [ "${ensure_prey_visible}" = "0" ]; then
  visibility_args="--disable_prey_visibility_guarantee"
  echo "prey visibility guarantee disabled"
fi

model_args=""
if [ -n "${model_dir}" ]; then
  model_args="--model_dir ${model_dir}"
  echo "loading checkpoint: ${model_dir}"
fi

wandb_args=""
if [ "${use_wandb}" = "True" ] || [ "${use_wandb}" = "true" ] || [ "${use_wandb}" = "1" ]; then
  wandb_args="--use_wandb True"
fi

python train/train_long_range_predator_prey.py \
 --seed "${seed}" \
 --truelyDistributed True \
 --env_name "${env}" \
 --algorithm_name "${algo}" \
 --experiment_name "${exp}" \
 --scenario "${scenario}" \
 --num_predators "${num_predators}" \
 --predator_max_speed "${predator_max_speed}" \
 --num_prey "${num_prey}" \
 --world_size "${world_size}" \
 --obs_radius "${obs_radius}" \
 --comm_radius "${comm_radius}" \
 ${graph_args} \
 ${visibility_args} \
 --capture_radius "${capture_radius}" \
 --capture_k "${capture_k}" \
 --prey_speed_ratio "${prey_speed_ratio}" \
 --collision_radius "${collision_radius}" \
 --env_device "${env_device}" \
 --faulty_node "${faulty_node}" \
 --eval_faulty_node "${eval_faulty_node}" \
 --iterations 3 \
 --gnn_loss_coef 1 \
 --critic_lr 5e-04 \
 --lr 5e-04 \
 --n_embd "${hidden_dim}" \
 --hidden_size "${hidden_dim}" \
 --out_channels "${hidden_dim}" \
 --value_loss_coef 1 \
 --max_grad_norm 0.8 \
 --eval_episodes 5 \
 --n_training_threads "${n_training_threads}" \
 --n_rollout_threads "${n_rollout_threads}" \
 --n_eval_rollout_threads "${n_eval_rollout_threads}" \
 --num_mini_batch 1 \
 --mini_batch_size 6400 \
 --episode_length "${episode_length}" \
 --env_episode_length "${env_episode_length}" \
 --eval_interval 25 \
 --num_env_steps 200000000 \
 --ppo_epoch 15 \
 --gamma 0.999 \
 --gae_lambda 0.95 \
 --entropy_coef 0.01 \
 --clip_param 0.2 \
 --add_center_xy \
 --use_state_agent \
 --use_eval \
 --n_quants 1 \
 --num-heads 1 \
 --num-layers 3 \
 --user_name "${user_name}" \
 ${model_args} \
 ${wandb_args}
