import io
import json
import os

import numpy as np
import torch
import webdataset as wds
from PIL import Image

from mat.runner.shared.mujoco_runner import MujocoRunner, _t2n, faulty_action


class MujocoRunnerWDS(MujocoRunner):
    def __init__(self, config):
        super().__init__(config)
        self._wds_writer = None
        self._wds_episode = 0
        self._wds_pattern = None
        self._wds_shard_size = None

    def _init_wds_writer(self):
        if self._wds_writer is not None:
            return
        out_dir = os.path.abspath(self.all_args.wds_out_dir)
        os.makedirs(out_dir, exist_ok=True)
        self._wds_pattern = os.path.join(out_dir, self.all_args.wds_shard_pattern)
        self._wds_shard_size = self.all_args.wds_shard_size
        self._wds_writer = wds.ShardWriter(self._wds_pattern, maxcount=self._wds_shard_size)

    def close_wds(self):
        if self._wds_writer is not None:
            self._wds_writer.close()
            self._wds_writer = None

    def _png_bytes(self, rgb):
        img = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _npy_bytes(self, array):
        buf = io.BytesIO()
        np.save(buf, array)
        return buf.getvalue()

    def _to_jsonable(self, value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return value.item()
        if isinstance(value, dict):
            return {k: self._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_jsonable(v) for v in value]
        return value

    def _write_sample(
        self,
        writer,
        episode_id,
        step_id,
        obs_np,
        share_obs_np,
        actions_np,
        env_actions_np,
        rewards_np,
        dones_np,
        infos_np,
    ):
        key = f"{episode_id:06d}_{step_id:06d}"
        frame = self.eval_envs.render(mode="rgb_array")[0]
        visibility = self.eval_envs.get_visibility_matrix()[0]
        edge_index = self.eval_envs.get_edge_index_matrix()[0]
        reward_mean = float(np.mean(rewards_np)) if rewards_np.size else 0.0
        visible_counts = visibility.sum(axis=1).tolist()
        caption = (
            f"episode {episode_id} step {step_id}; "
            f"reward_mean {reward_mean:.3f}; visible_counts {visible_counts}"
        )
        sample_info = {
            "episode_id": episode_id,
            "step_id": step_id,
            "done": bool(np.all(dones_np)),
            "reward_mean": reward_mean,
            "infos": self._to_jsonable(infos_np),
        }
        sample_info = self._to_jsonable(sample_info)
        writer.write(
            {
                "__key__": key,
                "image.png": self._png_bytes(frame),
                "caption.txt": caption,
                "obs.npy": self._npy_bytes(obs_np),
                "state.npy": self._npy_bytes(share_obs_np),
                "actions.npy": self._npy_bytes(actions_np),
                "env_actions.npy": self._npy_bytes(env_actions_np),
                "rewards.npy": self._npy_bytes(rewards_np),
                "dones.npy": self._npy_bytes(dones_np),
                "visibility.npy": self._npy_bytes(visibility.astype(np.int8)),
                "edge_index.npy": self._npy_bytes(edge_index.astype(np.int64)),
                "info.json": json.dumps(sample_info, sort_keys=True).encode("utf-8"),
            }
        )

    @torch.no_grad()
    def eval(self, total_num_steps, faulty_node):
        if self.all_args.n_eval_rollout_threads != 1:
            raise ValueError("collect_wds requires n_eval_rollout_threads=1 for render support.")

        eval_episode = 0
        eval_episode_rewards = []
        one_episode_rewards = [0 for _ in range(self.all_args.eval_episodes)]

        eval_obs, eval_share_obs, _ = self.eval_envs.reset()
        eval_obs = torch.tensor(eval_obs, dtype=torch.float32, device="cuda:0")
        eval_share_obs = torch.tensor(eval_share_obs, dtype=torch.float32, device="cuda:0")
        eval_rnn_states = torch.zeros(
            (self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.n_embd),
            dtype=torch.float32,
            device="cuda:0",
        )
        eval_masks = torch.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=torch.float32, device="cuda:0")

        self._init_wds_writer()
        writer = self._wds_writer

        episode_id = self._wds_episode
        step_id = 0
        record_reset = False
        action_dim = self.envs.action_space[0].shape[0]

        eval_obs_np = _t2n(eval_obs)
        eval_share_obs_np = _t2n(eval_share_obs)
        actions_np = np.full((self.num_agents, action_dim), np.nan, dtype=np.float32)
        env_actions_np = np.full_like(actions_np, np.nan)
        rewards_np = np.zeros((self.num_agents, 1), dtype=np.float32)
        dones_np = np.zeros((self.num_agents,), dtype=bool)
        infos_np = {"phase": "reset"}
        self._write_sample(
            writer,
            episode_id,
            step_id,
            eval_obs_np[0],
            eval_share_obs_np[0],
            actions_np,
            env_actions_np,
            rewards_np,
            dones_np,
            infos_np,
        )

        while True:
            if record_reset:
                eval_obs_np = _t2n(eval_obs)
                eval_share_obs_np = _t2n(eval_share_obs)
                actions_np = np.full((self.num_agents, action_dim), np.nan, dtype=np.float32)
                env_actions_np = np.full_like(actions_np, np.nan)
                rewards_np = np.zeros((self.num_agents, 1), dtype=np.float32)
                dones_np = np.zeros((self.num_agents,), dtype=bool)
                infos_np = {"phase": "reset"}
                self._write_sample(
                    writer,
                    episode_id,
                    step_id,
                    eval_obs_np[0],
                    eval_share_obs_np[0],
                    actions_np,
                    env_actions_np,
                    rewards_np,
                    dones_np,
                    infos_np,
                )
                record_reset = False

            batch_edge_index = None
            if self.all_args.iterations > 0:
                if self.algorithm_name in {"mappo_gnn", "mappo_dgnn", "mappo_dgnn_dsgd"}:
                    edge_index = self.eval_envs.get_edge_index_matrix()
                    edge_index = torch.tensor(edge_index, dtype=torch.float32, device="cuda:0")
                    batch_edge_index = self.get_batch_edge_index(edge_index)
                    x = self.trainer.policy.transformer.obs_encoder(eval_obs, batch_edge_index)
                    eval_obs = torch.cat([eval_obs, x], dim=-1).detach()

            self.trainer.prep_rollout()
            eval_actions, eval_rnn_states = self.trainer.policy.act(
                eval_share_obs,
                eval_obs,
                eval_rnn_states,
                eval_masks,
                None,
                deterministic=True,
                batched_edge_index=batch_edge_index,
            )
            eval_actions = eval_actions.reshape(self.n_eval_rollout_threads, self.num_agents, -1)
            eval_rnn_states = eval_rnn_states.reshape(self.n_eval_rollout_threads, self.num_agents, -1)

            eval_actions = faulty_action(eval_actions.cpu().detach().numpy(), faulty_node)
            env_actions = 100 * eval_actions
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, _ = self.eval_envs.step(env_actions)

            step_id += 1
            self._write_sample(
                writer,
                episode_id,
                step_id,
                eval_obs[0],
                eval_share_obs[0],
                eval_actions[0],
                env_actions[0],
                eval_rewards[0],
                eval_dones[0],
                eval_infos[0],
            )

            if self.all_args.use_render:
                self.eval_envs.render(mode="human")

            eval_rewards_mean = np.mean(eval_rewards, axis=1).flatten()
            one_episode_rewards += eval_rewards_mean

            eval_obs = torch.tensor(eval_obs, dtype=torch.float32, device="cuda:0")
            eval_share_obs = torch.tensor(eval_share_obs, dtype=torch.float32, device="cuda:0")
            eval_dones_tensor = torch.tensor(eval_dones, dtype=torch.float32, device="cuda:0")

            eval_dones_env = torch.all(eval_dones_tensor, dim=1)
            eval_rnn_states[eval_dones_env == True] = torch.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, self.n_embd),
                dtype=torch.float32,
                device="cuda:0",
            )
            eval_masks = torch.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=torch.float32, device="cuda:0")
            eval_masks[eval_dones_env == True] = torch.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=torch.float32, device="cuda:0")

            for eval_i in range(self.all_args.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    episode_id += 1
                    self._wds_episode = episode_id
                    step_id = 0
                    record_reset = True
                    eval_episode += 1
                    eval_episode_rewards.append(one_episode_rewards[eval_i])
                    one_episode_rewards[eval_i] = 0

            if eval_episode >= self.all_args.eval_episodes:
                key_average = f"faulty_node_{faulty_node}/eval_average_episode_rewards"
                key_max = f"faulty_node_{faulty_node}/eval_max_episode_rewards"
                eval_env_infos = {key_average: eval_episode_rewards,
                                  key_max: [np.max(eval_episode_rewards)]}

                self.log_env(eval_env_infos, total_num_steps)
                print("faulty_node {} eval_average_episode_rewards is {}."
                      .format(faulty_node, np.mean(eval_episode_rewards)))

                if self.use_wandb:
                    import wandb
                    wandb.log({"eval_average_episode_rewards": np.mean(eval_episode_rewards)}, step=total_num_steps)

                if self.reward_list is None:
                    self.reward_list = np.array(np.mean(eval_episode_rewards))
                else:
                    self.reward_list = np.hstack((self.reward_list, np.array(np.mean(eval_episode_rewards))))

                np.save(self.plt_name + "_eval.npy", self.reward_list)
                break

        self._wds_episode = episode_id
