#!/usr/bin/env python
"""Render LongRangePredatorPreyContinuous-v0 rollouts.

Example:
    python mat/scripts/render/render_long_range_predator_prey.py \
        --output predator_prey_rollout.gif --policy chase
"""

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mat.envs.long_range_predator_prey  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(description="Render the long-range predator-prey environment.")
    parser.add_argument("--output", type=str, default="predator_prey_rollout.gif")
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_predators", type=int, default=6)
    parser.add_argument("--num_prey", type=int, default=2)
    parser.add_argument("--episode_length", type=int, default=200)
    parser.add_argument("--world_size", type=float, default=6.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--policy", choices=["chase", "random", "zero"], default="chase")
    parser.add_argument("--random_action_scale", type=float, default=0.35)
    parser.add_argument("--save_every_frame", action="store_true")
    return parser.parse_args()


def scripted_chase_actions(env):
    """A simple debug controller using privileged state for visual sanity checks."""
    core = env.unwrapped.core
    pred = core.predator_pose[0, :, :2].detach().cpu().numpy()
    pred_theta = core.predator_pose[0, :, 2].detach().cpu().numpy()
    prey = core.prey_pose[0, :, :2].detach().cpu().numpy()
    alive = core.prey_alive[0].detach().cpu().numpy().astype(bool)

    actions = np.zeros((core.n_predators, 2), dtype=np.float32)
    if not alive.any():
        return actions

    live_prey = prey[alive]
    for i in range(core.n_predators):
        rel = live_prey - pred[i]
        target = rel[np.argmin(np.linalg.norm(rel, axis=-1))]
        desired = np.arctan2(target[1], target[0])
        err = np.arctan2(np.sin(desired - pred_theta[i]), np.cos(desired - pred_theta[i]))
        actions[i, 0] = 0.8
        actions[i, 1] = np.clip(err / np.pi, -1.0, 1.0)
    return actions


def make_actions(env, policy, scale):
    n_agents = env.unwrapped.n_agents
    if policy == "zero":
        return np.zeros((n_agents, 2), dtype=np.float32)
    if policy == "random":
        return np.random.uniform(-scale, scale, size=(n_agents, 2)).astype(np.float32)
    return scripted_chase_actions(env)


def main():
    args = parse_args()
    np.random.seed(args.seed)

    env = gym.make(
        "LongRangePredatorPreyContinuous-v0",
        num_predators=args.num_predators,
        num_prey=args.num_prey,
        episode_length=args.episode_length,
        world_size=args.world_size,
        device=args.device,
        disable_env_checker=True,
    )
    env.unwrapped.reset(seed=args.seed)

    frames = []
    for step in range(args.steps):
        actions = make_actions(env, args.policy, args.random_action_scale)
        _, _, rewards, dones, infos, _ = env.unwrapped.step(actions)
        frames.append(env.unwrapped.render(mode="rgb_array"))

        if step % max(args.fps, 1) == 0:
            remaining = infos[0]["prey_remaining"] if infos else "?"
            reward = float(np.mean(rewards))
            print(f"step={step:04d} reward={reward:.3f} prey_remaining={remaining}")

        if np.all(dones):
            print(f"episode finished at step {step}")
            break

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise SystemExit("imageio is required for saving renders. Install it with `pip install imageio`.") from exc

    if output.suffix.lower() == ".gif":
        imageio.mimsave(output, frames, fps=args.fps)
    elif output.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        imageio.imwrite(output, frames[-1])
    else:
        output.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            imageio.imwrite(output / f"frame_{i:04d}.png", frame)

    if args.save_every_frame and output.suffix.lower() != "":
        frame_dir = output.with_suffix("")
        frame_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            imageio.imwrite(frame_dir / f"frame_{i:04d}.png", frame)

    env.close()
    print(f"saved {output}")


if __name__ == "__main__":
    main()
