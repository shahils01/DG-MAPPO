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
sys.path.append("../../")
from mat.config import get_config
from mat.envs.starcraft2.StarCraft2_Env import StarCraft2Env
# from mat.envs.smacv2.smacv2.env.starcraft2.starcraft2 import StarCraft2Env_ as SMAC_v2
from mat.envs.starcraft2.Random_StarCraft2_Env import RandomStarCraft2Env
from mat.envs.env_wrappers import ShareSubprocVecEnv
from mat.envs.starcraft2.smac_maps import get_map_params
# from mat.envs.smacv2.smacv2.env.starcraft2.maps.smac_maps import get_map_params
# from mat.runner.shared.smac_runner import SMACRunner
from mat.runner.shared.smac_runner_new import SMACRunner
from mat.runner.shared.dgn_runner import DGNRunner
from mat.algorithms.mat.algorithm.dg_mat import resolve_agent_devices

yaml_path = os.path.join(os.path.expanduser("~"), "Desktop", "marl_ws", 
                        "Multi-Agent-Transformer", "mat", "envs", "smacv2",
                        "smacv2", "examples", "configs")

"""Train script for SMAC."""
def make_train_env(all_args, env_config=None):
    def get_env_fn(rank, env_config):
        def init_env():
            if all_args.env_name == "StarCraft2":
                if all_args.random_agent_order:
                    env = RandomStarCraft2Env(all_args)
                else:
                    env = StarCraft2Env(all_args)

                env.seed(all_args.seed + rank * 1000)

            # elif all_args.env_name == "smacv2":
            #     if env_config is not None:
            #         env = SMAC_v2(all_args, **env_config)
            #     else:
            #         env = SMAC_v2(all_args)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            # env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    # Keep SMAC in subprocesses even for a single rollout thread. An in-process
    # SC2 controller cannot be interrupted safely when its RPC call hangs.
    return ShareSubprocVecEnv(
        [get_env_fn(i, env_config) for i in range(all_args.n_rollout_threads)],
        timeout=all_args.smac_worker_timeout,
    )


def make_eval_env(all_args, env_config=None):
    def get_env_fn(rank, env_config):
        def init_env():
            if all_args.env_name == "StarCraft2":
                if all_args.random_agent_order:
                    env = RandomStarCraft2Env(all_args)
                else:
                    env = StarCraft2Env(all_args)
                
                env.seed(all_args.seed * 50000 + rank * 10000)
                
            # elif all_args.env_name == "smacv2":
            #     all_args.seed = all_args.seed * 50000 + rank * 10000
            #     if env_config is not None:
            #         env = SMAC_v2(all_args, **env_config)
            #     else:
            #         env = SMAC_v2(all_args)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            # env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    return ShareSubprocVecEnv(
        [get_env_fn(i, env_config) for i in range(all_args.n_eval_rollout_threads)],
        timeout=all_args.smac_worker_timeout,
    )


def parse_args(args, parser):
    parser.add_argument('--map_name', type=str, default='3m', help="Which smac map to run on")
    parser.add_argument('--eval_map_name', type=str, default='3m', help="Which smac map to eval on")
    parser.add_argument('--unit_sight_range', type=float, default=4.0,
                        help="Sight range used for allied and enemy unit observations")
    enemy_info_group = parser.add_mutually_exclusive_group()
    enemy_info_group.add_argument(
        "--share_enemy_info_with_neighbors",
        dest="share_enemy_info_with_neighbors",
        action="store_true",
        help=(
            "include enemies seen by direct communication neighbors in each "
            "agent observation (default)"
        ),
    )
    enemy_info_group.add_argument(
        "--disable_enemy_info_sharing",
        "--no_share_enemy_info_with_neighbors",
        dest="share_enemy_info_with_neighbors",
        action="store_false",
        help=(
            "restrict enemy features to enemies directly inside the observing "
            "agent's sight range"
        ),
    )
    parser.set_defaults(share_enemy_info_with_neighbors=True)
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
    parser.add_argument(
        "--smac_worker_timeout",
        type=float,
        default=120.0,
        help="Kill training if an SMAC worker does not respond within this many seconds",
    )

    all_args = parser.parse_known_args(args)[0]
    if all_args.unit_sight_range <= 0:
        parser.error("--unit_sight_range must be greater than 0")
    if all_args.smac_worker_timeout <= 0:
        parser.error("--smac_worker_timeout must be greater than 0")

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    print(
        "SMAC neighbor enemy-info sharing: "
        f"{'enabled' if all_args.share_enemy_info_with_neighbors else 'disabled'}"
    )

    # Keep the original DG-MAT flags as backward-compatible aliases while the
    # generic names also cover MAPPO-DGNN-DSGD.
    all_args.agent_parallel = bool(
        all_args.agent_parallel or all_args.dg_mat_agent_parallel
    )
    if all_args.agent_parallel_devices is None:
        all_args.agent_parallel_devices = all_args.dg_mat_devices

    if all_args.algorithm_name == "mat_dec":
        all_args.dec_actor = True
        all_args.share_actor = True
        all_args.truelyDistributed = False

    if all_args.algorithm_name == "dg_mat":
        # DG-MAT owns one graph-attention actor/critic pair per agent and uses
        # graph-neighbor D-SGD. It deliberately avoids MAT's broken zero-state
        # encode_state path.
        all_args.encode_state = False
        all_args.dec_actor = False
        all_args.share_actor = False
        all_args.share_policy = False
        all_args.truelyDistributed = True

    if all_args.agent_parallel and all_args.algorithm_name not in {
        "dg_mat",
        "mappo_dgnn_dsgd",
    }:
        parser.error(
            "--agent_parallel is supported only with dg_mat or mappo_dgnn_dsgd"
        )
    if all_args.agent_parallel:
        all_args.truelyDistributed = True

    if all_args.algorithm_name == "mat_gnn":
        all_args.strict_local_obs = False

    if all_args.algorithm_name in {"ippo", "consensus_ippo"}:
        all_args.iterations = 0
        all_args.truelyDistributed = False
        all_args.n_quants = 1
        if all_args.algorithm_name == "consensus_ippo":
            all_args.share_policy = False

    if all_args.algorithm_name == "dgn":
        all_args.iterations = 0
        all_args.n_quants = 1

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        primary_cuda_index = 0
        if all_args.agent_parallel and all_args.agent_parallel_devices:
            first_device = all_args.agent_parallel_devices.split(",", 1)[0].strip()
            if first_device.startswith("cuda:"):
                first_device = first_device.split(":", 1)[1]
            try:
                primary_cuda_index = int(first_device)
            except ValueError:
                parser.error(
                    "--agent_parallel_devices must contain CUDA indices such as 0,1 "
                    "or cuda:0,cuda:1"
                )
            if primary_cuda_index < 0:
                parser.error("--agent_parallel_devices CUDA indices must be non-negative")
        device = torch.device(f"cuda:{primary_cuda_index}")
        if all_args.agent_parallel:
            try:
                resolve_agent_devices(
                    primary_device=device,
                    n_agent=1,
                    enabled=True,
                    device_spec=all_args.agent_parallel_devices,
                )
            except ValueError as error:
                parser.error(str(error))
        torch.cuda.set_device(device)
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        if all_args.agent_parallel:
            parser.error("--agent_parallel requires CUDA and at least two visible GPUs")
        print("choose to use cpu...")
        device = torch.device("cpu")
        #torch.set_num_threads(all_args.n_training_threads)

    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results") / all_args.env_name / all_args.map_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
            all_args.user_name))

    
    env_config = {}
    env_config['env_args'] = None
    wandb_project = all_args.map_name
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = get_map_params(all_args.map_name)["n_agents"]
    all_args.run_dir = run_dir

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
                         project=wandb_project + "_journal_new",
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                              "ablation" +
                              "_seed" + str(all_args.seed),
                         group=all_args.experiment_name,
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

    runner_cls = DGNRunner if all_args.algorithm_name == "dgn" else SMACRunner
    runner = runner_cls(config)
    try:
        runner.run()
    finally:
        # These closes are bounded for subprocess environments, so cleanup
        # cannot hang forever after an SC2 server failure.
        envs.close()
        if eval_envs is not None and eval_envs is not envs:
            eval_envs.close()

        if all_args.use_wandb:
            run.finish()
        else:
            runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
            runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
