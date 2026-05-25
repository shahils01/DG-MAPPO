#!/usr/bin/env python
import sys
import os
import yaml
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
sys.path.append("../../../")
from mat.config import get_config
from mat.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv_turtleBot
from mat.runner.shared.turtleBot_runner import turtleBotRunner as Runner

import gymnasium as gym
import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train your RL agent.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_known_args()[0]

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print('Done launching app')
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
import isaaclab.sim as sim_utils

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import distMARL.tasks  # noqa: F401

yaml_path = os.path.join(os.path.expanduser("~"), "Desktop", "marl_ws", 
                        "Multi-Agent-Transformer", "mat", "envs", "smacv2",
                        "smacv2", "examples", "configs")

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.turtlebot3 import TURTLEBOT3_CFG

from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectMARLEnvCfg

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass


@configclass
class TurtleBot3MARLEnvCfg(DirectMARLEnvCfg):
    # Environment settings
    decimation = 2
    episode_length_s = 15.0  # 30 second episodes
    
    # Multi-agent specification
    num_agents = 3
    possible_agents = ["turtlebot_0", "turtlebot_1", "turtlebot_2"]
    
    # Action and observation spaces for each agent
    # Actions: [linear_velocity, angular_velocity] instead of wheel velocities
    action_spaces = {agent: 2 for agent in possible_agents}
    # Observations: [self_pos_x, self_pos_y, self_vel_x, self_vel_y, self_orientation, 
    #                goal_rel_x, goal_rel_y, other_bot1_rel_x, other_bot1_rel_y, 
    #                other_bot2_rel_x, other_bot2_rel_y]
    observation_spaces = {agent: 11 for agent in possible_agents}
    state_space = -1  # No global state

    # Simulation settings
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # Scene settings
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,  # Number of parallel environments
        env_spacing=6.0,
        replicate_physics=True
    )


    # Robot configurations (3 TurtleBots)
    turtlebot_0_cfg: ArticulationCfg = TURTLEBOT3_CFG.replace(
        prim_path="/World/envs/env_.*/TurtleBot_0"
    )
    turtlebot_1_cfg: ArticulationCfg = TURTLEBOT3_CFG.replace(
        prim_path="/World/envs/env_.*/TurtleBot_1"
    )
    turtlebot_2_cfg: ArticulationCfg = TURTLEBOT3_CFG.replace(
        prim_path="/World/envs/env_.*/TurtleBot_2"
    )

    # Workspace plane configuration (black plane at 0.001m height)
    workspace_plane_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/workspace_plane",
        spawn=sim_utils.CuboidCfg(
            size=(3.048, 3.048, 0.001),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.1, 0.1, 0.1),  # Black color
                metallic=0.0,
                roughness=0.8,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.001),
            rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    # Task region configurations (at height 0.0015m)
    task_region_size = (0.6096, 0.6096, 0.001)
    workspace_half_size = 3.048 / 2.0
    region_half_size = 0.6096 / 2.0
    
    # Start region (bottom-left corner)
    start_region_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/start_region",
        spawn=sim_utils.CuboidCfg(
            size=task_region_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 0.0),  # Green for start
                metallic=0.0,
                roughness=0.8,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-workspace_half_size + region_half_size, 
                 -workspace_half_size + region_half_size, 0.0015),
            rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    # Region A (top-left corner)
    region_a_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/region_a",
        spawn=sim_utils.CuboidCfg(
            size=task_region_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),  # Red for A
                metallic=0.0,
                roughness=0.8,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-workspace_half_size + region_half_size,
                 workspace_half_size - region_half_size, 0.0015),
            rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    # Region B (top-right corner)
    region_b_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/region_b",
        spawn=sim_utils.CuboidCfg(
            size=task_region_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.0, 1.0),  # Blue for B
                metallic=0.0,
                roughness=0.8,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(workspace_half_size - region_half_size,
                 workspace_half_size - region_half_size, 0.0015),
            rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    # Region C (bottom-right corner)
    region_c_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/region_c",
        spawn=sim_utils.CuboidCfg(
            size=task_region_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 1.0, 0.0),  # Yellow for C
                metallic=0.0,
                roughness=0.8,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(workspace_half_size - region_half_size,
                 -workspace_half_size + region_half_size, 0.0015),
            rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    # Task parameters
    task_region_positions = {
        "A": (-workspace_half_size + region_half_size, workspace_half_size - region_half_size),
        "B": (workspace_half_size - region_half_size, workspace_half_size - region_half_size),
        "C": (workspace_half_size - region_half_size, -workspace_half_size + region_half_size),
        "start": (0.0, 0.0)
    }
    
    # Robot parameters
    wheel_base = 0.160  # TurtleBot3 Burger wheel base (m)
    wheel_radius = 0.033  # TurtleBot3 Burger wheel radius (m)
    max_linear_vel = 0.22  # m/s - max linear velocity
    max_angular_vel = 2.84  # rad/s - max angular velocity (from TurtleBot3 specs)
    robot_radius = 0.105  # Approximate radius for collision checking
    
    # Simplified reward parameters
    rew_distance_scale = -1.0  # Scale for distance-based Huber loss
    rew_distance_huber_delta = 0.2  # Delta for Huber loss
    rew_collision_scale = -10.0  # Penalty for inter-robot collisions
    rew_collision_distance_threshold = 0.25  # m - distance threshold for collision penalty
    rew_angular_vel_scale = -0.1  # Penalty for large angular velocities
    rew_goal_reached_scale = 10.0  # Reward for reaching goal
    
    # Goal parameters
    goal_reached_threshold = 0.2  # m - distance to consider goal reached
    goal_assignment_mode = "random_unique"  # Each robot gets unique goal
    stop_at_goal = True  # Robot stops when it reaches its goal


"""Train script for SMAC."""
def make_train_env(all_args, env_config=None):
    def get_env_fn(rank, env_config):
        def init_env():
            if all_args.env_name == "multi_turtleBot3":
                env = gym.make(all_args.task, cfg=env_config, render_mode="rgb_array" if all_args.video else None)
                env.seed(all_args.seed + rank * 1000)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            # env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv_turtleBot([get_env_fn(0, env_config)])
    else:
        return ShareDummyVecEnv_turtleBot([get_env_fn(0, env_config)])


def make_eval_env(all_args, env_config=None):
    def get_env_fn(rank, env_config):
        def init_env():
            if all_args.env_name == "multi_turtleBot3":
                env = gym.make(all_args.task, cfg=env_config, render_mode="rgb_array" if all_args.video else None)
                env.seed(all_args.seed*5000 + rank * 1000)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            # env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv_turtleBot([get_env_fn(0, env_config)])
    else:
        return ShareDummyVecEnv_turtleBot([get_env_fn(0, env_config)])


def parse_args(args, parser):
    parser.add_argument('--run_dir', type=str, default='', help="Which smac map to eval on")
    parser.add_argument("--add_move_state", action='store_true', default=False)
    parser.add_argument("--add_local_obs", action='store_true', default=False)
    parser.add_argument("--add_distance_state", action='store_true', default=False)
    parser.add_argument("--add_enemy_action_state", action='store_true', default=False)
    parser.add_argument("--add_agent_id", action='store_true', default=False)
    parser.add_argument("--add_visible_state", action='store_true', default=False)
    parser.add_argument("--add_xy_state", action='store_true', default=False)
    parser.add_argument("--use_state_agent", action='store_false', default=True)
    parser.add_argument("--use_mustalive", action='store_false', default=True)
    parser.add_argument("--add_center_xy", action='store_false', default=True)
    parser.add_argument("--random_agent_order", action='store_true', default=False)
    parser.add_argument("--strict_local_obs", type=bool, default=False)

    # arguments specific to issac-lab
    parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
    parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
    parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
    parser.add_argument("--num_envs", type=int, default=32, help="Number of environments to simulate.")
    parser.add_argument("--num_agents", type=int, default=3, help="Number of agents to simulate.")
    parser.add_argument("--task", type=str, default="Turtlebot-v0", help="Name of the task.")
    parser.add_argument(
        "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
    parser.add_argument("--max_iterations", type=int, default=1000000, help="RL Policy training iterations.")
    parser.add_argument(
        "--ml_framework",
        type=str,
        default="torch",
        choices=["torch", "jax", "jax-numpy"],
        help="The ML framework used for training the skrl agent.",
    )

    all_args = parser.parse_known_args(args)[0]

    return all_args

# parser = get_config()
# all_args = parse_args(sys.argv[1:], parser)

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # Initialize env_cfg properly using your task configuration
    env_cfg = TurtleBot3MARLEnvCfg()
    # if getattr(env_cfg, "scene", None) is None or not hasattr(env_cfg.scene, "num_envs"):
    #     from isaaclab.envs import SceneCfg  # or the correct config class
    #     env_cfg.scene = SceneCfg()          # initialize with defaults or your values

    if all_args.algorithm_name == "mat_dec":
        all_args.dec_actor = True
        all_args.share_actor = True

    if all_args.algorithm_name == "mat_gnn":
        all_args.strict_local_obs = False

    # cuda
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

    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results") / all_args.env_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
            all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = all_args.num_envs if all_args.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = device
    env_cfg.sim.device = f"cuda:0"

    wandb_project = all_args.env_name

    envs = make_train_env(all_args, env_cfg)
    eval_envs = make_eval_env(all_args, env_cfg) if all_args.use_eval else None
    num_agents = envs.n_agents
    
    all_args.run_dir = run_dir
    all_args.iterations = num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }
    
    print('config = ', config)

    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=wandb_project,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                              str(all_args.experiment_name) +
                              "_seed" + str(all_args.seed),
                         group=all_args.map_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                             str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])