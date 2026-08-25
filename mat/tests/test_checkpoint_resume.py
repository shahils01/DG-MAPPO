"""Regression coverage for full update-boundary checkpoint resumption."""

from pathlib import Path
import random
import sys
from types import SimpleNamespace
import types
import tempfile
import unittest

import numpy as np
import torch

# The lightweight local test environment does not install the optional W&B
# client used by cluster runs.
if "wandb" not in sys.modules:
    wandb_stub = types.ModuleType("wandb")
    wandb_stub.run = None
    sys.modules["wandb"] = wandb_stub

from mat.runner.shared.base_runner import Runner
from mat.utils.checkpointing import (
    checkpoint_directory,
    checkpoint_metadata,
    resolve_resume_checkpoint,
)


class _Policy:
    def __init__(self):
        self.transformer = torch.nn.Linear(2, 1)
        self.optimizers = [torch.optim.Adam(self.transformer.parameters(), lr=0.01)]


class _CheckpointRunner(Runner):
    def checkpoint_runner_state(self):
        return {"counter": self.counter}

    def restore_runner_state(self, state):
        self.counter = state.get("counter", 0)


def _runner(checkpoint_dir):
    runner = _CheckpointRunner.__new__(_CheckpointRunner)
    runner.policy = _Policy()
    runner.trainer = SimpleNamespace(value_normalizer=torch.nn.Linear(1, 1))
    runner.algorithm_name = "mappo_dgnn_dsgd"
    runner.num_agents = 2
    runner.episode_length = 10
    runner.n_rollout_threads = 4
    runner.checkpoint_dir = Path(checkpoint_dir)
    runner.use_wandb = False
    runner.start_episode = 0
    runner.resumed_total_num_steps = 0
    runner.counter = 17
    return runner


class CheckpointResumeTest(unittest.TestCase):
    def test_full_checkpoint_restores_training_and_rng_state(self):
        with tempfile.TemporaryDirectory() as directory:
            source = _runner(directory)
            loss = source.policy.transformer(torch.ones(1, 2)).sum()
            loss.backward()
            source.policy.optimizers[0].step()
            source.trainer.value_normalizer.weight.data.fill_(3.0)

            random.seed(31)
            np.random.seed(32)
            torch.manual_seed(33)
            source.save_checkpoint(episode=7)

            expected_python = random.random()
            expected_numpy = np.random.rand()
            expected_torch = torch.rand(3)
            saved_model = {
                key: value.detach().clone()
                for key, value in source.policy.transformer.state_dict().items()
            }

            restored = _runner(directory)
            restored.counter = -1
            restored.restore_checkpoint(Path(directory) / "latest.pt")

            self.assertEqual(restored.start_episode, 8)
            self.assertEqual(restored.resumed_total_num_steps, 320)
            self.assertEqual(restored.counter, 17)
            self.assertEqual(random.random(), expected_python)
            self.assertEqual(np.random.rand(), expected_numpy)
            self.assertTrue(torch.equal(torch.rand(3), expected_torch))
            for key, value in restored.policy.transformer.state_dict().items():
                self.assertTrue(torch.equal(value, saved_model[key]))
            self.assertTrue(
                torch.equal(
                    restored.trainer.value_normalizer.weight,
                    torch.full_like(
                        restored.trainer.value_normalizer.weight, 3.0
                    ),
                )
            )
            optimizer_state = restored.policy.optimizers[0].state_dict()["state"]
            self.assertTrue(optimizer_state)

    def test_auto_resume_uses_stable_seed_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                seed=4,
                checkpoint_dir=None,
                resume_checkpoint=None,
                auto_resume=True,
            )
            expected_dir = Path(directory) / "checkpoints" / "seed4"
            self.assertEqual(checkpoint_directory(args, directory), expected_dir)
            self.assertIsNone(resolve_resume_checkpoint(args, directory))

            expected_dir.mkdir(parents=True)
            runner = _runner(expected_dir)
            runner.save_checkpoint(episode=2)
            resolved = resolve_resume_checkpoint(args, directory)
            self.assertEqual(resolved, expected_dir / "latest.pt")
            metadata = checkpoint_metadata(resolved)
            self.assertEqual(metadata["algorithm_name"], "mappo_dgnn_dsgd")
            self.assertEqual(metadata["next_episode"], 3)
            self.assertEqual(metadata["total_num_steps"], 120)


if __name__ == "__main__":
    unittest.main()
