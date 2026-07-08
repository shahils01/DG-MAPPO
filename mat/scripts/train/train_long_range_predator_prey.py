#!/usr/bin/env python
"""Train DG-MAPPO-style algorithms on LongRangePredatorPreyContinuous-v0."""

import os
import socket
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

try:
    import setproctitle
except ImportError:
    setproctitle = None

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mat.envs.long_range_predator_prey  # noqa: F401
from mat.config import get_config
from mat.envs.env_wrappers import ShareDummyVecEnv, ShareSubprocVecEnv


def make_env_kwargs(all_args, seed):
    return {
        "num_predators": all_args.num_predators,
        "num_prey": all_args.num_prey,
        "world_size": all_args.world_size,
        "dt": all_args.env_dt,
        "episode_length": all_args.episode_length,
        "obs_radius": all_args.obs_radius,
        "comm_radius": all_args.comm_radius,
        "capture_radius": all_args.capture_radius,
        "capture_k": all_args.capture_k,
        "predator_max_speed": all_args.predator_max_speed,
        "predator_max_omega": all_args.predator_max_omega,
        "prey_speed_ratio": all_args.prey_speed_ratio,
        "prey_max_omega": all_args.prey_max_omega,
        "prey_avoid_radius": all_args.prey_avoid_radius,
        "collision_radius": all_args.collision_radius,
        "collision_resolution_iters": all_args.collision_resolution_iters,
        "device": all_args.env_device,
        "seed": seed,
    }


def seed_env(env, seed):
    target = getattr(env, "unwrapped", env)
    if hasattr(target, "seed"):
        target.seed(seed)
    else:
        env.reset(seed=seed)


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name != "long_range_predator_prey":
                raise NotImplementedError(f"Unsupported env_name: {all_args.env_name}")
            env = gym.make(
                all_args.scenario,
                disable_env_checker=True,
                **make_env_kwargs(all_args, all_args.seed + rank * 1000),
            )
            seed_env(env, all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name != "long_range_predator_prey":
                raise NotImplementedError(f"Unsupported env_name: {all_args.env_name}")
            env = gym.make(
                all_args.scenario,
                disable_env_checker=True,
                **make_env_kwargs(all_args, all_args.seed * 50000 + rank * 10000),
            )
            seed_env(env, all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument("--scenario", type=str, default="LongRangePredatorPreyContinuous-v0")
    parser.add_argument("--num_predators", type=int, default=6)
    parser.add_argument("--num_prey", type=int, default=2)
    parser.add_argument("--world_size", type=float, default=6.0)
    parser.add_argument("--env_dt", type=float, default=0.1)
    parser.add_argument("--obs_radius", type=float, default=1.8)
    parser.add_argument("--comm_radius", type=float, default=2.2)
    parser.add_argument("--capture_radius", type=float, default=0.35)
    parser.add_argument("--capture_k", type=int, default=2)
    parser.add_argument("--predator_max_speed", type=float, default=0.22)
    parser.add_argument("--predator_max_omega", type=float, default=2.84)
    parser.add_argument("--prey_speed_ratio", type=float, default=0.85)
    parser.add_argument("--prey_max_omega", type=float, default=2.3)
    parser.add_argument("--prey_avoid_radius", type=float, default=2.4)
    parser.add_argument("--collision_radius", type=float, default=0.18)
    parser.add_argument("--collision_resolution_iters", type=int, default=4)
    parser.add_argument("--env_device", type=str, default="cpu")
    parser.add_argument("--faulty_node", type=int, default=-1)
    parser.add_argument("--eval_faulty_node", type=int, nargs="+", default=[-1])
    parser.add_argument("--add_center_xy", action="store_true", default=False)
    parser.add_argument("--use_state_agent", action="store_true", default=False)
    parser.add_argument("--use_mustalive", action="store_false", default=True)

    all_args = parser.parse_known_args(args)[0]
    if all_args.eval_faulty_node is None:
        all_args.eval_faulty_node = [-1]
    return all_args


def configure_algorithm(all_args):
    if all_args.algorithm_name == "mat_dec":
        all_args.dec_actor = True
        all_args.share_actor = True
        all_args.truelyDistributed = False

    if all_args.algorithm_name in {"ippo", "consensus_ippo"}:
        all_args.iterations = 0
        all_args.truelyDistributed = False
        all_args.n_quants = 1
        if all_args.algorithm_name == "consensus_ippo":
            all_args.share_policy = False

    if all_args.algorithm_name == "dgn":
        all_args.iterations = 0
        all_args.n_quants = 1


def main(args):
    import wandb

    from mat.runner.shared.dgn_runner import DGNRunner
    from mat.runner.shared.ma_gotogoal_runner import MAGoToGoalRunner

    parser = get_config()
    all_args = parse_args(args, parser)
    configure_algorithm(all_args)

    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    print("predator-prey config:", all_args)

    run_dir = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
        / all_args.env_name
        / all_args.scenario
        / f"{all_args.num_predators}pred_{all_args.num_prey}prey"
        / all_args.algorithm_name
        / all_args.experiment_name
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    if all_args.use_wandb:
        run = wandb.init(
            config=all_args,
            project=all_args.scenario,
            entity=all_args.user_name,
            notes=socket.gethostname(),
            name=f"{all_args.algorithm_name}_{all_args.experiment_name}_seed{all_args.seed}",
            group=all_args.env_name,
            dir=str(run_dir),
            job_type="training",
            reinit=True,
        )
    else:
        exst_run_nums = [
            int(str(folder.name).split("run")[1])
            for folder in run_dir.iterdir()
            if str(folder.name).startswith("run")
        ]
        curr_run = "run1" if len(exst_run_nums) == 0 else f"run{max(exst_run_nums) + 1}"
        run_dir = run_dir / curr_run
        run_dir.mkdir(parents=True, exist_ok=True)
        run = None

    if setproctitle is not None:
        setproctitle.setproctitle(
            f"{all_args.algorithm_name}-{all_args.env_name}-{all_args.experiment_name}@{all_args.user_name}"
        )

    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = envs.n_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    runner_cls = DGNRunner if all_args.algorithm_name == "dgn" else MAGoToGoalRunner
    runner = runner_cls(config)
    runner.run()

    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
