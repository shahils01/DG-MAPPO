#!/bin/sh

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "${SCRIPT_DIR}" || exit 1

env="long_range_predator_prey"
scenario="LongRangePredatorPreyContinuous-v0"

# Example:
#   MODEL_DIR=results/long_range_predator_prey/.../models/transformer_10.pt sh eval_predator_prey.sh
#   REWARD_MODE=local USE_REWARD_CONSENSUS=True REWARD_CONSENSUS_STEPS=3 sh eval_predator_prey.sh
# If MODEL_DIR is unset, evaluation uses random initialized weights and prints a warning.
algo="${ALGO:-mappo_dgnn_dsgd}"
exp="${EXP:-eval}"
seed="${SEED:-1}"
model_dir="${MODEL_DIR:-/home/shahils/Desktop/gitBackupRepo/DG-MAPPO/mat/scripts/results/long_range_predator_prey/LongRangePredatorPreyContinuous-v0/6pred_2prey/mappo_dgnn_dsgd/single/wandb/run-20260710_015425-zntft6p3/files/transformer_4300.pt}"

num_predators="${NUM_PREDATORS:-6}"
predator_max_speed="${PREDATOR_MAX_SPEED:-${predator_max_speed:-0.7}}"
num_prey="${NUM_PREY:-2}"
world_size="${WORLD_SIZE:-6.0}"
random_start_positions="${RANDOM_START_POSITIONS:-True}"
init_min_predator_dist="${INIT_MIN_PREDATOR_DIST:-0.45}"
init_min_prey_dist="${INIT_MIN_PREY_DIST:-0.45}"
init_min_prey_predator_dist="${INIT_MIN_PREY_PREDATOR_DIST:-1.0}"
obs_radius="${OBS_RADIUS:-1.8}"
comm_radius="${COMM_RADIUS:-2.2}"
ensure_connected_comm_graph="${ENSURE_CONNECTED_COMM_GRAPH:-True}"
ensure_prey_visible="${ENSURE_PREY_VISIBLE:-True}"
capture_radius="${CAPTURE_RADIUS:-0.35}"
capture_k="${CAPTURE_K:-1}"
prey_speed_ratio="${PREY_SPEED_RATIO:-0.85}"
collision_radius="${COLLISION_RADIUS:-0.18}"
reward_mode="${REWARD_MODE:-global}"
use_reward_consensus="${USE_REWARD_CONSENSUS:-False}"
reward_consensus_steps="${REWARD_CONSENSUS_STEPS:-3}"
env_device="${ENV_DEVICE:-cpu}"

hidden_dim="${HIDDEN_DIM:-128}"
eval_episodes="${EVAL_EPISODES:-5}"
episode_length="${EPISODE_LENGTH:-200}"
env_episode_length="${ENV_EPISODE_LENGTH:-${episode_length}}"
render_mode="${RENDER_MODE:-gif}"
render_fps="${RENDER_FPS:-10}"
output_dir="${EVAL_OUTPUT_DIR:-}"
gif_prefix="${GIF_PREFIX:-eval}"
user_name="${USER_NAME:-xxx}"

echo "env=${env}, scenario=${scenario}, algo=${algo}, exp=${exp}, seed=${seed}"
echo "num_predators=${num_predators}, num_prey=${num_prey}, comm_radius=${comm_radius}, obs_radius=${obs_radius}"
echo "render_mode=${render_mode}, render_fps=${render_fps}"
echo "rollout episode_length=${episode_length}, env_episode_length=${env_episode_length}"

random_start_args=""
if [ "${random_start_positions}" = "True" ] || [ "${random_start_positions}" = "true" ] || [ "${random_start_positions}" = "1" ]; then
  random_start_args="--random_start_positions"
  echo "fully random start positions enabled"
fi

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

reward_consensus_args=""
if [ "${use_reward_consensus}" = "True" ] || [ "${use_reward_consensus}" = "true" ] || [ "${use_reward_consensus}" = "1" ]; then
  reward_consensus_args="--use_reward_consensus"
  echo "reward consensus enabled for ${reward_consensus_steps} rounds"
fi

model_args=""
if [ -n "${model_dir}" ]; then
  model_args="--model_dir ${model_dir}"
else
  echo "WARNING: MODEL_DIR is not set. Eval will use a randomly initialized policy."
fi

output_args=""
if [ -n "${output_dir}" ]; then
  output_args="--eval_output_dir ${output_dir}"
fi

python eval/eval_long_range_predator_prey.py \
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
 ${random_start_args} \
 --init_min_predator_dist "${init_min_predator_dist}" \
 --init_min_prey_dist "${init_min_prey_dist}" \
 --init_min_prey_predator_dist "${init_min_prey_predator_dist}" \
 --obs_radius "${obs_radius}" \
 --comm_radius "${comm_radius}" \
 ${graph_args} \
 ${visibility_args} \
 --capture_radius "${capture_radius}" \
 --capture_k "${capture_k}" \
 --prey_speed_ratio "${prey_speed_ratio}" \
 --collision_radius "${collision_radius}" \
 --reward_mode "${reward_mode}" \
 ${reward_consensus_args} \
 --reward_consensus_steps "${reward_consensus_steps}" \
 --env_device "${env_device}" \
 --iterations 3 \
 --gnn_loss_coef 10 \
 --critic_lr 5e-04 \
 --lr 5e-04 \
 --n_embd "${hidden_dim}" \
 --hidden_size "${hidden_dim}" \
 --out_channels "${hidden_dim}" \
 --value_loss_coef 1 \
 --max_grad_norm 0.8 \
 --eval_episodes "${eval_episodes}" \
 --n_training_threads 1 \
 --n_rollout_threads 1 \
 --n_eval_rollout_threads 1 \
 --num_mini_batch 1 \
 --mini_batch_size 4000 \
 --episode_length "${episode_length}" \
 --env_episode_length "${env_episode_length}" \
 --ppo_epoch 10 \
 --gamma 0.99 \
 --gae_lambda 0.95 \
 --entropy_coef 0.01 \
 --clip_param 0.2 \
 --add_center_xy \
 --use_state_agent \
 --n_quants 1 \
 --num-heads 1 \
 --num-layers 3 \
 --render_mode "${render_mode}" \
 --render_fps "${render_fps}" \
 --eval_gif_prefix "${gif_prefix}" \
 --user_name "${user_name}" \
 ${model_args} \
 ${output_args}
