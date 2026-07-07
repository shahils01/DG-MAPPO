import os
import time

import numpy as np
import torch
import wandb
from tensorboardX import SummaryWriter

from mat.algorithms.dgn.dgn_trainer import DGNTrainer
from mat.utils.util import get_shape_from_obs_space


def _as_scalar(value):
    if torch.is_tensor(value):
        return value.detach().cpu().item()
    return float(value)


def faulty_action(action, faulty_node):
    action_fault = action.copy()
    if faulty_node >= 0:
        action_fault[:, faulty_node, :] = 0.0
    return action_fault


class DGNRunner:
    def __init__(self, config):
        self.all_args = config["all_args"]
        self.envs = config["envs"]
        self.eval_envs = config["eval_envs"]
        self.device = config["device"]
        self.num_agents = config["num_agents"]
        self.run_dir = config["run_dir"]
        self.use_wandb = self.all_args.use_wandb
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.eval_episodes = self.all_args.eval_episodes
        self.save_interval = self.all_args.save_interval
        self.log_interval = self.all_args.log_interval
        self.episode_length = self.all_args.episode_length
        self.num_env_steps = self.all_args.num_env_steps
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.obs_dim = get_shape_from_obs_space(self.envs.observation_space[0])[0]
        self.action_space = self.envs.action_space[0]
        self.action_type = "Continuous" if self.action_space.__class__.__name__ == "Box" else "Discrete"

        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
        else:
            self.log_dir = str(self.run_dir / "logs")
            os.makedirs(self.log_dir, exist_ok=True)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / "models")
            os.makedirs(self.save_dir, exist_ok=True)

        self.trainer = DGNTrainer(
            self.all_args,
            self.envs.observation_space[0],
            self.envs.action_space[0],
            self.num_agents,
            self.device,
        )

        if self.all_args.model_dir is not None:
            self.trainer.restore(self.all_args.model_dir)

    def current_adj(self, envs):
        adj = envs.get_visibility_matrix()
        adj = np.asarray(adj, dtype=np.float32)
        if adj.shape[-1] != self.num_agents:
            adj = adj[:, :, :self.num_agents]
        return adj

    def normalize_available_actions(self, available_actions):
        if self.action_type != "Discrete" or available_actions is None:
            return None
        arr = np.asarray(available_actions)
        if arr.dtype == object or arr.ndim < 3:
            return None
        return arr.astype(np.float32)

    def run(self):
        obs, _, available_actions = self.envs.reset()
        obs = np.asarray(obs, dtype=np.float32)
        available_actions = self.normalize_available_actions(available_actions)
        adj = self.current_adj(self.envs)

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        train_episode_rewards = np.zeros(self.n_rollout_threads, dtype=np.float32)
        done_episode_rewards = []
        latest_train_info = self.trainer.empty_info()

        for episode in range(episodes):
            for _ in range(self.episode_length):
                deterministic = self.trainer.total_env_steps < self.all_args.dgn_warmup_steps
                if deterministic and self.action_type == "Discrete":
                    actions = self.random_discrete_actions(available_actions)
                elif deterministic and self.action_type == "Continuous":
                    actions = self.random_continuous_actions()
                else:
                    actions = self.trainer.select_actions(obs, adj, available_actions, deterministic=False)

                env_actions = actions
                if self.action_type == "Continuous":
                    env_actions = faulty_action(env_actions, self.all_args.faulty_node)

                next_obs, _, rewards, dones, infos, next_available_actions = self.envs.step(env_actions)
                next_obs = np.asarray(next_obs, dtype=np.float32)
                rewards = np.asarray(rewards, dtype=np.float32)
                dones = np.asarray(dones, dtype=np.float32)
                next_available_actions = self.normalize_available_actions(next_available_actions)
                next_adj = self.current_adj(self.envs)

                self.trainer.store(
                    obs,
                    actions,
                    rewards,
                    dones,
                    next_obs,
                    adj,
                    next_adj,
                    available_actions,
                    next_available_actions,
                )

                for _ in range(self.all_args.dgn_updates_per_step):
                    latest_train_info = self.trainer.train()

                dones_env = np.all(dones, axis=1)
                train_episode_rewards += rewards.mean(axis=1).reshape(-1)
                for env_i, done in enumerate(dones_env):
                    if done:
                        done_episode_rewards.append(train_episode_rewards[env_i])
                        train_episode_rewards[env_i] = 0.0

                obs = next_obs
                available_actions = next_available_actions
                adj = next_adj

            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save(episode)

            if episode % self.log_interval == 0:
                end = time.time()
                print(
                    f"Algo {self.algorithm_name} Exp {self.experiment_name} updates {episode}/{episodes}, "
                    f"timesteps {total_num_steps}/{self.num_env_steps}, FPS {int(total_num_steps / (end - start))}."
                )
                log_info = {k: _as_scalar(v) for k, v in latest_train_info.items()}
                if done_episode_rewards:
                    log_info["average_episode_rewards"] = float(np.mean(done_episode_rewards))
                    done_episode_rewards = []
                self.log_train(log_info, total_num_steps)

            if self.use_eval and episode % self.eval_interval == 0:
                if self.action_type == "Continuous":
                    faulty_nodes = self.all_args.eval_faulty_node or [-1]
                    for node in faulty_nodes:
                        self.eval(total_num_steps, node)
                else:
                    self.eval(total_num_steps)

    def random_discrete_actions(self, available_actions):
        actions = np.random.randint(0, self.action_space.n, size=(self.n_rollout_threads, self.num_agents, 1))
        if available_actions is not None:
            for env_i in range(self.n_rollout_threads):
                for agent_i in range(self.num_agents):
                    valid = np.flatnonzero(available_actions[env_i, agent_i] > 0)
                    if valid.size > 0:
                        actions[env_i, agent_i, 0] = np.random.choice(valid)
        return actions

    def random_continuous_actions(self):
        low = self.action_space.low.reshape(1, 1, -1)
        high = self.action_space.high.reshape(1, 1, -1)
        return np.random.uniform(low, high, size=(self.n_rollout_threads, self.num_agents, self.action_space.shape[0])).astype(np.float32)

    @torch.no_grad()
    def eval(self, total_num_steps, faulty_node=-1):
        if self.eval_envs is None:
            return

        obs, _, available_actions = self.eval_envs.reset()
        obs = np.asarray(obs, dtype=np.float32)
        available_actions = self.normalize_available_actions(available_actions)
        adj = self.current_adj(self.eval_envs)
        eval_episode = 0
        episode_rewards = np.zeros(self.n_eval_rollout_threads, dtype=np.float32)
        completed_rewards = []

        while eval_episode < self.eval_episodes:
            actions = self.trainer.select_actions(obs, adj, available_actions, deterministic=True)
            if self.action_type == "Continuous":
                actions = faulty_action(actions, faulty_node)

            obs, _, rewards, dones, infos, available_actions = self.eval_envs.step(actions)
            obs = np.asarray(obs, dtype=np.float32)
            rewards = np.asarray(rewards, dtype=np.float32)
            dones = np.asarray(dones, dtype=np.float32)
            available_actions = self.normalize_available_actions(available_actions)
            adj = self.current_adj(self.eval_envs)

            episode_rewards += rewards.mean(axis=1).reshape(-1)
            dones_env = np.all(dones, axis=1)
            for env_i, done in enumerate(dones_env):
                if done:
                    eval_episode += 1
                    completed_rewards.append(episode_rewards[env_i])
                    episode_rewards[env_i] = 0.0

        key = "eval_average_episode_rewards" if faulty_node < 0 else f"faulty_node_{faulty_node}/eval_average_episode_rewards"
        value = float(np.mean(completed_rewards)) if completed_rewards else 0.0
        self.log_train({key: value}, total_num_steps)
        print(f"{key} is {value}.")

    def log_train(self, train_infos, total_num_steps):
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    def save(self, episode):
        self.trainer.save(self.save_dir, episode)
