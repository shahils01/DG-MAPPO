import wandb
import os
import random
import shutil
from pathlib import Path
import numpy as np
import torch
from mat.utils.shared_buffer import SharedReplayBuffer
from mat.algorithms.mat.mat_trainer import MATTrainer as TrainAlgo
from mat.algorithms.mat.algorithm.transformer_policy import TransformerPolicy as Policy
from mat.utils.logging import SummaryWriter
from mat.utils.util import get_shape_from_obs_space
from mat.utils.checkpointing import (
    LATEST_CHECKPOINT_NAME,
    checkpoint_directory,
    load_training_checkpoint,
    resolve_resume_checkpoint,
)

def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()

class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']       

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        # self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N
        self.n_embd = self.all_args.n_embd
        
        act_space = self.envs.action_space[0]

        if act_space.__class__.__name__ == 'Box':
            self.action_type = 'Continuous'
        else:
            self.action_type = 'Discrete'

        self.obs_dim = get_shape_from_obs_space(self.envs.observation_space[0])[0]
        if self.action_type == 'Discrete':
            self.act_dim = act_space.n
        else:
            self.act_dim = act_space.shape[0]
        
        self.num_quants = self.all_args.n_quants

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir
        self.checkpoint_dir = checkpoint_directory(
            self.all_args, config["run_dir"]
        )
        self.resume_checkpoint = resolve_resume_checkpoint(
            self.all_args, config["run_dir"]
        )
        if self.model_dir is not None and self.resume_checkpoint is not None:
            raise ValueError(
                "Use either --model_dir for a weight-only warm start or "
                "--resume_checkpoint/--auto_resume for full resumption, not both."
            )
        self.start_episode = 0
        self.resumed_total_num_steps = 0

        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        if (
            self.algorithm_name.startswith('mat')
            or self.algorithm_name == "mappo_dgnn_dsgd"
        ):
            share_observation_space = (
                self.envs.share_observation_space[0]
                if self.use_centralized_V
                else self.envs.observation_space[0]
            )
        else:
            share_observation_space = self.envs.share_observation_space[0]
            
        # policy network
        self.policy = Policy(self.all_args,
                             self.envs.observation_space[0],
                             share_observation_space,
                             self.envs.action_space[0],
                             self.num_agents,
                             device=self.device)

        # algorithm
        self.trainer = TrainAlgo(self.all_args, self.policy, self.num_agents, device=self.device)

        # Optimizers and value normalization are created by the trainer, so a
        # full training checkpoint must be restored after trainer creation.
        if self.resume_checkpoint is not None:
            self.restore_checkpoint(self.resume_checkpoint)
        elif self.model_dir is not None:
            self.restore(self.model_dir)
        
        # buffer
        self.buffer = SharedReplayBuffer(self.all_args,
                                        self.num_agents,
                                        self.envs.observation_space[0],
                                        share_observation_space,
                                        self.envs.action_space[0],
                                         self.all_args.env_name,
                                         training_device=self.device)
        if getattr(self.policy.transformer, "agent_parallel_enabled", False):
            owners = [str(device) for device in self.policy.transformer.agent_devices]
            print(f"[agent parallel] agent owners={owners}")
        print(
            "[rollout buffer] "
            f"storage={self.buffer.storage_device}, "
            f"pinned_memory={self.buffer.pin_memory}"
        )

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self, adjacency_matrix=None, batched_edge_index=None):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        if self.buffer.available_actions is None:
            next_values = self.trainer.policy.get_values(
                self.buffer.share_obs[-1],
                self.buffer.obs[-1],
                self.buffer.rnn_states_critic[-1],
                self.buffer.masks[-1],
                adjacency_matrix=adjacency_matrix,
                batched_edge_index=batched_edge_index,
            )
        else:
            next_values = self.trainer.policy.get_values(
                self.buffer.share_obs[-1],
                self.buffer.obs[-1],
                self.buffer.rnn_states_critic[-1],
                self.buffer.masks[-1],
                self.buffer.available_actions[-1],
                adjacency_matrix=adjacency_matrix,
                batched_edge_index=batched_edge_index,
            )

        # action_log, next_values, _ = self.trainer.policy.transformer(state, obs, action, available_actions, action_hats)

        next_values = next_values.reshape(self.n_rollout_threads, self.num_agents, -1)

        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)
    
    def train(self, episode):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer, episode, self.obs_dim)      
        self.buffer.after_update()
        return train_infos

    def save(self, episode):
        """Save legacy model weights and a resumable training checkpoint."""
        self.save_checkpoint(episode)
        self.policy.save(self.save_dir, episode)

    @staticmethod
    def _cpu_state(value):
        """Recursively detach tensors so checkpoints are device portable."""
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {key: Runner._cpu_state(item) for key, item in value.items()}
        if isinstance(value, list):
            return [Runner._cpu_state(item) for item in value]
        if isinstance(value, tuple):
            return tuple(Runner._cpu_state(item) for item in value)
        return value

    def _optimizers(self):
        optimizers = self.policy.optimizers
        return optimizers if isinstance(optimizers, (list, tuple)) else [optimizers]

    def checkpoint_runner_state(self):
        """Hook for small runner-specific counters."""
        return {}

    def restore_runner_state(self, state):
        """Restore runner-specific counters saved by ``checkpoint_runner_state``."""

    def save_checkpoint(self, episode):
        """Atomically save all state needed to continue at the next update."""
        checkpoint_dir = Path(self.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        total_num_steps = (
            (int(episode) + 1)
            * self.episode_length
            * self.n_rollout_threads
        )
        value_normalizer = self.trainer.value_normalizer
        checkpoint = {
            "checkpoint_version": 1,
            "algorithm_name": self.algorithm_name,
            "num_agents": self.num_agents,
            "episode": int(episode),
            "next_episode": int(episode) + 1,
            "total_num_steps": total_num_steps,
            "model_state_dict": self._cpu_state(
                self.policy.transformer.state_dict()
            ),
            "optimizer_state_dicts": [
                self._cpu_state(optimizer.state_dict())
                for optimizer in self._optimizers()
            ],
            "value_normalizer_state_dict": (
                self._cpu_state(value_normalizer.state_dict())
                if value_normalizer is not None
                else None
            ),
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": (
                    [state.cpu() for state in torch.cuda.get_rng_state_all()]
                    if torch.cuda.is_available()
                    else None
                ),
            },
            "runner_state": self.checkpoint_runner_state(),
            "wandb_run_id": (
                wandb.run.id
                if self.use_wandb and wandb.run is not None
                else None
            ),
        }

        checkpoint_path = checkpoint_dir / f"checkpoint_{int(episode)}.pt"
        temporary_path = checkpoint_dir / (
            f".{checkpoint_path.name}.{os.getpid()}.tmp"
        )
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, checkpoint_path)

        latest = checkpoint_dir / LATEST_CHECKPOINT_NAME
        latest_temporary = checkpoint_dir / f".{LATEST_CHECKPOINT_NAME}.{os.getpid()}.tmp"
        try:
            latest_temporary.unlink(missing_ok=True)
            latest_temporary.symlink_to(checkpoint_path.name)
            os.replace(latest_temporary, latest)
        except OSError:
            latest_temporary.unlink(missing_ok=True)
            shutil.copy2(checkpoint_path, latest_temporary)
            os.replace(latest_temporary, latest)

        print(
            f"Saved resumable checkpoint at update {episode}: {checkpoint_path}"
        )

    def restore_checkpoint(self, checkpoint_path):
        """Restore complete training state and continue at the next update."""
        checkpoint = load_training_checkpoint(checkpoint_path)
        checkpoint_algorithm = checkpoint.get("algorithm_name")
        if checkpoint_algorithm not in (None, self.algorithm_name):
            raise ValueError(
                "Checkpoint algorithm mismatch: "
                f"checkpoint={checkpoint_algorithm}, current={self.algorithm_name}"
            )
        checkpoint_agents = checkpoint.get("num_agents")
        if checkpoint_agents not in (None, self.num_agents):
            raise ValueError(
                "Checkpoint agent-count mismatch: "
                f"checkpoint={checkpoint_agents}, current={self.num_agents}"
            )

        self.policy.transformer.load_state_dict(checkpoint["model_state_dict"])
        optimizer_states = checkpoint.get("optimizer_state_dicts", [])
        optimizers = self._optimizers()
        if len(optimizer_states) != len(optimizers):
            raise ValueError(
                "Checkpoint optimizer-count mismatch: "
                f"checkpoint={len(optimizer_states)}, current={len(optimizers)}"
            )
        for optimizer, state in zip(optimizers, optimizer_states):
            optimizer.load_state_dict(state)

        value_normalizer_state = checkpoint.get("value_normalizer_state_dict")
        if value_normalizer_state is not None:
            if self.trainer.value_normalizer is None:
                raise ValueError(
                    "Checkpoint contains value-normalizer state but the current "
                    "configuration has value normalization disabled."
                )
            self.trainer.value_normalizer.load_state_dict(value_normalizer_state)

        rng_state = checkpoint.get("rng_state", {})
        if rng_state.get("python") is not None:
            random.setstate(rng_state["python"])
        if rng_state.get("numpy") is not None:
            np.random.set_state(rng_state["numpy"])
        if rng_state.get("torch") is not None:
            torch.set_rng_state(rng_state["torch"])
        cuda_rng_state = rng_state.get("cuda")
        if cuda_rng_state is not None and torch.cuda.is_available():
            if len(cuda_rng_state) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(cuda_rng_state)
            else:
                print(
                    "Skipping CUDA RNG restoration because the checkpoint and "
                    "current CUDA device counts differ."
                )

        self.start_episode = int(checkpoint.get("next_episode", 0))
        self.resumed_total_num_steps = int(
            checkpoint.get(
                "total_num_steps",
                self.start_episode
                * self.episode_length
                * self.n_rollout_threads,
            )
        )
        self.restore_runner_state(checkpoint.get("runner_state", {}))
        print(
            f"Resumed complete training state from {checkpoint_path}; "
            f"continuing at update {self.start_episode} "
            f"({self.resumed_total_num_steps} environment steps)."
        )

    def restore(self, model_dir):
        """Restore policy's networks from a saved model."""
        allow_partial_restore = getattr(self.all_args, "allow_partial_restore", False)
        self.policy.restore(model_dir, allow_partial=allow_partial_restore)
 
    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v)>0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
