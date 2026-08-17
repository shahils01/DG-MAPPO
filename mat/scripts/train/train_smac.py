#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
sys.path.append("../../")
from mat.config import get_config
from mat.envs.smacv2_adapter import SMACv2EnvAdapter, load_smacv2_env_args
from mat.envs.env_wrappers import ShareSubprocVecEnv
from mat.runner.shared.smac_runner_new import SMACRunner
from mat.runner.shared.dgn_runner import DGNRunner
from mat.algorithms.mat.algorithm.dg_mat import resolve_agent_devices

"""Train script for SMAC."""
def make_train_env(all_args, env_config=None):
    def get_env_fn(rank, env_config):
        def init_env():
            if all_args.env_name.lower() == "starcraft2":
                # Import legacy SMAC only for legacy runs. Importing it before
                # SMACv2 registers duplicate PySC2 map names such as ``3m``.
                from mat.envs.starcraft2.StarCraft2_Env import StarCraft2Env
                from mat.envs.starcraft2.Random_StarCraft2_Env import (
                    RandomStarCraft2Env,
                )

                if all_args.random_agent_order:
                    env = RandomStarCraft2Env(all_args)
                else:
                    env = StarCraft2Env(all_args)

                env.seed(all_args.seed + rank * 1000)

            elif all_args.env_name.lower() == "smacv2":
                env = SMACv2EnvAdapter(
                    all_args,
                    env_args=env_config,
                    seed=all_args.seed + rank * 1000,
                )
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
            if all_args.env_name.lower() == "starcraft2":
                # Keep the legacy map registry out of SMACv2 worker processes.
                from mat.envs.starcraft2.StarCraft2_Env import StarCraft2Env
                from mat.envs.starcraft2.Random_StarCraft2_Env import (
                    RandomStarCraft2Env,
                )

                if all_args.random_agent_order:
                    env = RandomStarCraft2Env(all_args)
                else:
                    env = StarCraft2Env(all_args)
                
                env.seed(all_args.seed * 50000 + rank * 10000)
                
            elif all_args.env_name.lower() == "smacv2":
                env = SMACv2EnvAdapter(
                    all_args,
                    env_args=env_config,
                    seed=all_args.seed * 50000 + rank * 10000,
                )
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
    parser.add_argument(
        "--smacv2_config",
        type=str,
        default="terran_epo",
        help=(
            "SMACv2 YAML path or bundled alias: terran[_epo], protoss[_epo], "
            "or zerg[_epo] (default: terran_epo)"
        ),
    )
    parser.add_argument(
        "--smacv2_n_units",
        type=int,
        default=None,
        help="override capability_config.n_units from the SMACv2 YAML",
    )
    parser.add_argument(
        "--smacv2_n_enemies",
        type=int,
        default=None,
        help="override capability_config.n_enemies from the SMACv2 YAML",
    )
    parser.add_argument(
        "--smacv2_prob_obs_enemy",
        type=float,
        default=None,
        help="override the EPO enemy-observation probability (0 is strongest EPO)",
    )
    smacv2_action_mask_group = parser.add_mutually_exclusive_group()
    smacv2_action_mask_group.add_argument(
        "--smacv2_action_mask",
        dest="smacv2_action_mask",
        action="store_true",
        help="enable upstream target-action availability masks",
    )
    smacv2_action_mask_group.add_argument(
        "--smacv2_no_action_mask",
        dest="smacv2_action_mask",
        action="store_false",
        help="disable target-action masks so they cannot leak hidden enemies",
    )
    parser.set_defaults(smacv2_action_mask=None)
    parser.add_argument(
        "--smacv2_comm_range",
        type=float,
        default=None,
        help=(
            "fixed ally communication radius for graph algorithms; by default "
            "each pair uses the smaller of its SMACv2 unit sight ranges"
        ),
    )
    parser.add_argument(
        "--smacv2_force_connected_graph",
        action="store_true",
        default=False,
        help="repair disconnected ally graphs with minimum-distance edges",
    )
    parser.add_argument('--unit_sight_range', type=float, default=4.0,
                        help=(
                            "ally communication/visibility radius; enemy "
                            "observation keeps the standard SMAC range of 9"
                        ))
    enemy_info_group = parser.add_mutually_exclusive_group()
    enemy_info_group.add_argument(
        "--share_enemy_info_with_neighbors",
        dest="share_enemy_info_with_neighbors",
        action="store_true",
        help=(
            "include enemies seen by direct communication neighbors in each "
            "agent observation"
        ),
    )
    enemy_info_group.add_argument(
        "--disable_enemy_info_sharing",
        "--no_share_enemy_info_with_neighbors",
        dest="share_enemy_info_with_neighbors",
        action="store_false",
        help=(
            "restrict enemy features to enemies directly inside the observing "
            "agent's sight range (default)"
        ),
    )
    parser.set_defaults(share_enemy_info_with_neighbors=False)
    attack_visibility_group = parser.add_mutually_exclusive_group()
    attack_visibility_group.add_argument(
        "--strict_attack_visibility",
        dest="strict_attack_visibility",
        action="store_true",
        help=(
            "make targeted actions available only when the target is both in "
            "range and directly visible (default)"
        ),
    )
    attack_visibility_group.add_argument(
        "--allow_out_of_sight_attacks",
        "--disable_strict_attack_visibility",
        dest="strict_attack_visibility",
        action="store_false",
        help=(
            "restore legacy action masks that expose targets inside shooting "
            "range even when they are outside sight range"
        ),
    )
    parser.set_defaults(strict_attack_visibility=True)
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
    local_obs_group = parser.add_mutually_exclusive_group()
    local_obs_group.add_argument(
        "--strict_local_obs",
        dest="strict_local_obs",
        action="store_true",
        help="omit rich communication-neighbor ally features (default)",
    )
    local_obs_group.add_argument(
        "--include_communicating_ally_features",
        "--disable_strict_local_obs",
        dest="strict_local_obs",
        action="store_false",
        help="restore rich position, health, type, and last-action ally features",
    )
    parser.set_defaults(strict_local_obs=True)
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
    if all_args.smacv2_comm_range is not None and all_args.smacv2_comm_range <= 0:
        parser.error("--smacv2_comm_range must be greater than 0")

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    smacv2_env_args = None
    if all_args.env_name.lower() == "smacv2":
        try:
            smacv2_env_args, scenario_name, config_path = load_smacv2_env_args(
                config_name=all_args.smacv2_config,
                n_units=all_args.smacv2_n_units,
                n_enemies=all_args.smacv2_n_enemies,
                prob_obs_enemy=all_args.smacv2_prob_obs_enemy,
                action_mask=all_args.smacv2_action_mask,
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
            parser.error(str(error))
        all_args.map_name = scenario_name
        all_args.eval_map_name = scenario_name
        all_args.smacv2_map_name = smacv2_env_args["map_name"]
        print(f"SMACv2 config: {config_path}")
        print(
            "SMACv2 scenario: "
            f"{scenario_name}, map={all_args.smacv2_map_name}, "
            f"prob_obs_enemy={smacv2_env_args.get('prob_obs_enemy', 1.0)}, "
            f"action_mask={smacv2_env_args.get('action_mask', True)}, "
            f"force_connected_graph={all_args.smacv2_force_connected_graph}"
        )
    else:
        print(
            "SMAC neighbor enemy-info sharing: "
            f"{'enabled' if all_args.share_enemy_info_with_neighbors else 'disabled'}"
        )
        print(
            "SMAC strict attack visibility: "
            f"{'enabled' if all_args.strict_attack_visibility else 'disabled'}"
        )
        print(
            "SMAC strict local ally observations: "
            f"{'enabled' if all_args.strict_local_obs else 'disabled'}"
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

    if all_args.algorithm_name == "mappo_dgnn_dsgd":
        # D-SGD requires distinct agent-owned parameter sets and optimizers.
        # Make the algorithm name sufficient to select that execution mode.
        all_args.share_policy = False
        all_args.truelyDistributed = True
        # This method is decentralized by construction.  Neither the critic
        # nor its recurrent state may consume the global SMAC state.
        all_args.use_centralized_critic = False
        all_args.use_centralized_V = False
        all_args.encode_state = False

    if all_args.algorithm_name == "mappo":
        # Canonical homogeneous-agent MAPPO: shared local actor and
        # centralized state-value critic, with no graph or consensus terms.
        all_args.share_policy = False
        all_args.truelyDistributed = False
        all_args.iterations = 0
        all_args.n_quants = 1
        all_args.use_centralized_V = True
        all_args.use_centralized_critic = True
        all_args.encode_state = False

    recurrent_dgnn = all_args.use_actor_gru or all_args.use_critic_gru
    recurrent_algorithms = {"mappo", "mappo_dgnn_dsgd", "ippo", "consensus_ippo"}
    if recurrent_dgnn and all_args.algorithm_name not in recurrent_algorithms:
        parser.error(
            "--use_actor_gru and --use_critic_gru are currently supported "
            "only with mappo, mappo_dgnn_dsgd, ippo, or consensus_ippo"
        )
    if recurrent_dgnn:
        if all_args.recurrent_N != 1:
            parser.error("GRU policies currently require --recurrent_N 1")
        if all_args.data_chunk_length <= 0:
            parser.error("--data_chunk_length must be greater than zero")
        if all_args.episode_length % all_args.data_chunk_length != 0:
            parser.error(
                "--episode_length must be divisible by --data_chunk_length "
                "when a DGNN GRU is enabled"
            )
        all_args.use_recurrent_policy = True

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

    # A scheduler run should be able to direct artifacts to scratch storage.
    # Keep the historical repository-relative location when no override is
    # supplied so existing launches remain unchanged.
    if all_args.run_dir:
        run_dir = Path(all_args.run_dir).expanduser()
    else:
        run_dir = (
            Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])
            / "results"
            / all_args.env_name
            / all_args.map_name
            / all_args.algorithm_name
            / all_args.experiment_name
        )
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
            all_args.user_name))

    
    wandb_project = all_args.map_name
    envs = make_train_env(all_args, smacv2_env_args)
    eval_envs = (
        make_eval_env(all_args, smacv2_env_args) if all_args.use_eval else None
    )
    num_agents = envs.n_agents
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
