#!/usr/bin/env python
"""Evaluate and render LongRangePredatorPreyContinuous-v0 policies."""

import csv
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mat.config import get_config
from mat.scripts.train.train_long_range_predator_prey import (
    configure_algorithm,
    make_eval_env,
    optional_wandb,
    parse_args,
)

optional_wandb(False)

from mat.algorithms.dgn.dgn_trainer import DGNTrainer
from mat.algorithms.mat.algorithm.transformer_policy import TransformerPolicy


GNN_ALGORITHMS = {"mappo_gnn", "mappo_dgnn", "mappo_dgnn_dsgd"}


def parse_eval_args(args):
    parser = get_config()
    parser.add_argument("--eval_output_dir", type=str, default=None)
    parser.add_argument("--eval_gif_prefix", type=str, default="eval")
    parser.add_argument("--render_mode", choices=["gif", "human", "none"], default="gif")
    parser.add_argument("--render_fps", type=int, default=10)
    parser.add_argument("--save_render", action="store_true", default=True)
    parser.add_argument("--no_save_render", action="store_false", dest="save_render")
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--stochastic", action="store_false", dest="deterministic")
    all_args = parse_args(args, parser)
    configure_algorithm(all_args)
    all_args.use_eval = True
    all_args.n_eval_rollout_threads = 1
    all_args.n_rollout_threads = 1
    all_args.use_wandb = False
    if not all_args.save_render:
        all_args.render_mode = "none"
    all_args.save_render = all_args.render_mode == "gif"
    return all_args


def make_run_dir(all_args):
    default_dir = (
        REPO_ROOT
        / "mat"
        / "scripts"
        / "results"
        / all_args.env_name
        / all_args.scenario
        / f"{all_args.num_predators}pred_{all_args.num_prey}prey"
        / all_args.algorithm_name
        / all_args.experiment_name
        / "eval"
    )
    run_dir = Path(all_args.eval_output_dir) if all_args.eval_output_dir else default_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_gif(output, frames, fps):
    if not frames:
        return
    duration_ms = int(round(1000.0 / max(fps, 1)))
    images = [Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB") for frame in frames]
    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )
    with Image.open(output) as image:
        image.verify()


class HumanRenderer:
    def __init__(self, fps):
        self.fps = max(int(fps), 1)
        self.backend = None
        self.cv2 = None
        self.plt = None
        self.fig = None
        self.ax = None
        self.image = None
        self.warned = False

    def _init_backend(self, frame):
        try:
            import cv2

            self.cv2 = cv2
            cv2.namedWindow("LongRangePredatorPreyContinuous-v0", cv2.WINDOW_NORMAL)
            self.backend = "cv2"
            return
        except Exception as exc:
            cv2_error = exc

        try:
            import matplotlib.pyplot as plt

            self.plt = plt
            plt.ion()
            self.fig, self.ax = plt.subplots()
            self.image = self.ax.imshow(frame)
            self.ax.set_axis_off()
            self.fig.canvas.manager.set_window_title("LongRangePredatorPreyContinuous-v0")
            self.backend = "matplotlib"
            return
        except Exception as exc:
            if not self.warned:
                warnings.warn(
                    "Could not open a human render window. "
                    f"OpenCV error: {cv2_error}; matplotlib error: {exc}. "
                    "Use RENDER_MODE=gif on headless machines.",
                    RuntimeWarning,
                )
                self.warned = True
            self.backend = "disabled"

    def show(self, frame):
        frame = np.asarray(frame, dtype=np.uint8)
        if self.backend is None:
            self._init_backend(frame)

        if self.backend == "cv2":
            bgr = self.cv2.cvtColor(frame, self.cv2.COLOR_RGB2BGR)
            self.cv2.imshow("LongRangePredatorPreyContinuous-v0", bgr)
            delay_ms = max(int(round(1000.0 / self.fps)), 1)
            key = self.cv2.waitKey(delay_ms) & 0xFF
            return key not in (27, ord("q"))

        if self.backend == "matplotlib":
            self.image.set_data(frame)
            self.fig.canvas.draw_idle()
            self.plt.pause(1.0 / self.fps)
            return self.plt.fignum_exists(self.fig.number)

        return True

    def close(self):
        if self.backend == "cv2" and self.cv2 is not None:
            self.cv2.destroyWindow("LongRangePredatorPreyContinuous-v0")
        elif self.backend == "matplotlib" and self.plt is not None and self.fig is not None:
            self.plt.close(self.fig)


def first_info(infos):
    if infos is None:
        return {}
    info = infos[0] if isinstance(infos, (list, tuple)) and infos else infos
    if isinstance(info, (list, tuple)):
        info = info[0] if info else {}
    return info if isinstance(info, dict) else {}


def get_batch_edge_index(edge_index, num_agents, device):
    edge_index = torch.as_tensor(edge_index, dtype=torch.float32, device=device)
    batched_edges = []
    for env_i in range(edge_index.size(0)):
        valid_mask = edge_index[env_i, 1, :] != -1
        valid_edges = edge_index[env_i, :, valid_mask].clone()
        valid_edges[0, :] += env_i * num_agents
        valid_edges[1, :] += env_i * num_agents
        batched_edges.append(valid_edges)
    if not batched_edges:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.cat(batched_edges, dim=1).long()


def maybe_restore_mat(policy, model_dir):
    if model_dir is None:
        warnings.warn(
            "No --model_dir checkpoint was provided. Evaluating a randomly initialized policy. "
            "Pass MODEL_DIR=/path/to/transformer_*.pt to eval_predator_prey.sh for meaningful results.",
            RuntimeWarning,
        )
        return False
    policy.restore(model_dir, allow_partial=True)
    print(f"loaded checkpoint: {model_dir}")
    return True


def maybe_restore_dgn(trainer, model_dir):
    if model_dir is None:
        warnings.warn(
            "No --model_dir checkpoint was provided. Evaluating a randomly initialized DGN policy. "
            "Pass MODEL_DIR=/path/to/dgn_*.pt to eval_predator_prey.sh for meaningful results.",
            RuntimeWarning,
        )
        return False
    trainer.restore(model_dir)
    print(f"loaded checkpoint: {model_dir}")
    return True


def make_mat_policy(all_args, envs, device):
    obs_space = envs.observation_space[0]
    share_obs_space = envs.share_observation_space[0]
    if not all_args.use_centralized_V and all_args.algorithm_name.startswith("mat"):
        share_obs_space = obs_space
    policy = TransformerPolicy(
        all_args,
        obs_space,
        share_obs_space,
        envs.action_space[0],
        envs.n_agents,
        device=device,
    )
    maybe_restore_mat(policy, all_args.model_dir)
    policy.eval()
    return policy


def mat_actions(policy, all_args, envs, obs, share_obs, rnn_states, masks, device):
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    share_obs_t = torch.as_tensor(share_obs, dtype=torch.float32, device=device)
    if not all_args.use_centralized_V and all_args.algorithm_name.startswith("mat"):
        share_obs_t = obs_t

    batched_edge_index = None
    if all_args.iterations > 0 and all_args.algorithm_name in GNN_ALGORITHMS:
        edge_index = envs.get_edge_index_matrix()
        batched_edge_index = get_batch_edge_index(edge_index, envs.n_agents, device)
        encoded_obs = policy.transformer.obs_encoder(obs_t, batched_edge_index)
        obs_t = torch.cat([obs_t, encoded_obs], dim=-1).detach()

    actions, rnn_states = policy.act(
        share_obs_t,
        obs_t,
        rnn_states,
        masks,
        available_actions=None,
        deterministic=all_args.deterministic,
        batched_edge_index=batched_edge_index,
    )
    actions = actions.reshape(1, envs.n_agents, -1).detach().cpu().numpy()
    rnn_states = rnn_states.reshape(1, envs.n_agents, -1).detach()
    return actions, rnn_states


def dgn_actions(trainer, envs, obs, available_actions, deterministic):
    obs_np = np.asarray(obs, dtype=np.float32)
    adj = np.asarray(envs.get_visibility_matrix(), dtype=np.float32)
    actions = trainer.select_actions(obs_np, adj, available_actions, deterministic=deterministic)
    return actions


def rollout_episode(all_args, envs, actor, device, episode_idx, run_dir, human_renderer=None):
    obs, share_obs, available_actions = envs.reset()
    episode_reward = 0.0
    frames = []
    rnn_states = torch.zeros(
        (1, envs.n_agents, all_args.recurrent_N, all_args.n_embd),
        dtype=torch.float32,
        device=device,
    )
    masks = torch.ones((1, envs.n_agents, 1), dtype=torch.float32, device=device)

    for step in range(all_args.env_episode_length):
        if all_args.render_mode == "gif":
            frames.append(envs.render(mode="rgb_array")[0])
        elif all_args.render_mode == "human":
            frame = envs.render(mode="rgb_array")[0]
            if human_renderer is not None and not human_renderer.show(frame):
                break

        if all_args.algorithm_name == "dgn":
            actions = dgn_actions(actor, envs, obs, available_actions, all_args.deterministic)
        else:
            actions, rnn_states = mat_actions(actor, all_args, envs, obs, share_obs, rnn_states, masks, device)

        obs, share_obs, rewards, dones, infos, available_actions = envs.step(actions)
        episode_reward += float(np.asarray(rewards).mean())
        done_env = bool(np.all(dones))
        masks[:] = 0.0 if done_env else 1.0

        if done_env:
            if all_args.render_mode == "gif":
                frames.append(envs.render(mode="rgb_array")[0])
            elif all_args.render_mode == "human":
                frame = envs.render(mode="rgb_array")[0]
                if human_renderer is not None:
                    human_renderer.show(frame)
            break

    info = first_info(infos)
    if all_args.render_mode == "gif":
        gif_path = run_dir / f"{all_args.eval_gif_prefix}_episode_{episode_idx:03d}.gif"
        save_gif(gif_path, frames, all_args.render_fps)
    else:
        gif_path = None

    return {
        "episode": episode_idx,
        "reward": episode_reward,
        "steps": step + 1,
        "prey_remaining": info.get("prey_remaining", ""),
        "captures": info.get("captures", ""),
        "collision_count": info.get("collision_count", ""),
        "gif": str(gif_path) if gif_path is not None else "",
    }


def write_metrics(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    all_args = parse_eval_args(args)
    optional_wandb(False)

    if all_args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        print("choose to use gpu...")
    else:
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)
        print("choose to use cpu...")

    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    run_dir = make_run_dir(all_args)
    envs = make_eval_env(all_args)
    print("predator-prey eval config:", all_args)
    print(f"eval outputs: {run_dir}")

    if all_args.algorithm_name == "dgn":
        actor = DGNTrainer(all_args, envs.observation_space[0], envs.action_space[0], envs.n_agents, device)
        maybe_restore_dgn(actor, all_args.model_dir)
        actor.prep_rollout()
    else:
        actor = make_mat_policy(all_args, envs, device)

    rows = []
    human_renderer = HumanRenderer(all_args.render_fps) if all_args.render_mode == "human" else None
    try:
        for episode_idx in range(1, all_args.eval_episodes + 1):
            row = rollout_episode(
                all_args,
                envs,
                actor,
                device,
                episode_idx,
                run_dir,
                human_renderer=human_renderer,
            )
            rows.append(row)
            print(
                f"episode={row['episode']} reward={row['reward']:.3f} steps={row['steps']} "
                f"prey_remaining={row['prey_remaining']} gif={row['gif']}"
            )
    finally:
        if human_renderer is not None:
            human_renderer.close()
        envs.close()

    metrics_path = run_dir / "eval_metrics.csv"
    write_metrics(metrics_path, rows)
    rewards = [row["reward"] for row in rows]
    print(f"mean_reward={float(np.mean(rewards)) if rewards else 0.0:.3f}")
    print(f"wrote metrics: {metrics_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
