import sys
import types
import unittest

import torch
import torch.nn as nn

from mat.algorithms.mat.algorithm.transformer_policy import TransformerPolicy
from mat.algorithms.mat.mat_trainer import MATTrainer
from mat.config import get_config


def install_pyg_stubs_if_missing():
    """Provide the small PyG surface needed by CPU-only unit tests."""
    try:
        import torch_geometric  # noqa: F401
        return
    except ImportError:
        pass

    modules = {
        name: types.ModuleType(name)
        for name in (
            "torch_geometric",
            "torch_geometric.nn",
            "torch_geometric.nn.dense",
            "torch_geometric.nn.dense.linear",
            "torch_geometric.nn.inits",
            "torch_geometric.typing",
            "torch_geometric.data",
            "torch_scatter",
        )
    }

    class MessagePassing(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    class Linear(nn.Linear):
        def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            weight_initializer=None,
        ):
            del weight_initializer
            super().__init__(in_features, out_features, bias=bias)

    def glorot(value):
        parameters = value.parameters() if isinstance(value, nn.Module) else value
        for parameter in parameters:
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def ones(value):
        for parameter in value:
            nn.init.ones_(parameter)

    def scatter_add(src, index, dim=0, out=None, dim_size=None):
        if out is None:
            shape = list(src.shape)
            shape[dim] = dim_size
            out = torch.zeros(shape, dtype=src.dtype, device=src.device)
        return out.index_add(dim, index, src)

    modules["torch_geometric.nn"].MessagePassing = MessagePassing
    modules["torch_geometric.nn"].GATv2Conv = nn.Identity
    modules["torch_geometric.nn.dense.linear"].Linear = Linear
    modules["torch_geometric.nn.inits"].glorot = glorot
    modules["torch_geometric.nn.inits"].ones = ones
    modules["torch_geometric.typing"].OptTensor = torch.Tensor
    modules["torch_geometric.data"].Data = object
    modules["torch_scatter"].scatter_add = scatter_add
    sys.modules.update(modules)


class Box:
    def __init__(self, shape):
        self.shape = shape


class Discrete:
    def __init__(self, n):
        self.n = n


class MAPPOAgentParallelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_pyg_stubs_if_missing()

    def make_policy(self):
        args = get_config().parse_args([])
        overrides = {
            "algorithm_name": "mappo_dgnn_dsgd",
            "env_name": "StarCraft2",
            "iterations": 2,
            "num_layers": 2,
            "num_heads": 1,
            "n_embd": 16,
            "n_block": 1,
            "n_head": 1,
            "n_quants": 1,
            "truelyDistributed": True,
            "agent_parallel": False,
            "dg_mat_agent_parallel": False,
            "use_valuenorm": False,
            "consensusLoss": True,
            "detach": True,
        }
        for name, value in overrides.items():
            setattr(args, name, value)
        policy = TransformerPolicy(
            args,
            Box((7,)),
            Box((9,)),
            Discrete(5),
            3,
            torch.device("cpu"),
        )
        return args, policy

    @staticmethod
    def batched_edges():
        return torch.tensor(
            [
                [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
                [0, 1, 0, 2, 1, 2, 3, 4, 3, 5, 4, 5],
            ]
        )

    def test_rollout_ppo_and_parameter_ownership(self):
        _, policy = self.make_policy()
        raw_obs = torch.randn(2, 3, 7)
        with torch.no_grad():
            encoded = policy.transformer.obs_encoder(
                raw_obs, self.batched_edges()
            )
        obs = torch.cat([raw_obs, encoded], dim=-1)
        state = torch.randn(2, 3, 9)

        actions, rollout_log_probs, rollout_values = (
            policy.transformer.get_actions(state, obs, deterministic=True)
        )
        ppo_log_probs, ppo_values, entropy = policy.transformer(
            state, obs, actions
        )
        torch.testing.assert_close(rollout_log_probs, ppo_log_probs)
        torch.testing.assert_close(rollout_values, ppo_values)
        self.assertEqual(entropy.shape, (2, 3, 1))

        owned = set().union(
            *(
                {id(parameter) for parameter in policy.agent_parameters(agent_id)}
                for agent_id in range(3)
            )
        )
        self.assertEqual(
            owned, {id(parameter) for parameter in policy.transformer.parameters()}
        )

    def test_agent_parallel_ppo_update_and_dsgd(self):
        args, policy = self.make_policy()
        # Exercise the multi-device branches on CPU; CUDA placement itself is
        # covered by the shared round-robin resolver tests.
        policy.transformer.agent_parallel_enabled = True
        trainer = MATTrainer(args, policy, 3, torch.device("cpu"))

        batch_size, num_agents = 2, 3
        obs = torch.randn(batch_size, num_agents, 23)
        state = torch.randn(batch_size, num_agents, 9)
        adjacency = torch.ones(batch_size, num_agents, num_agents)
        with torch.no_grad():
            actions, old_log_probs, values = policy.transformer.get_actions(
                state, obs
            )

        sample = (
            state.reshape(batch_size * num_agents, 9),
            obs.reshape(batch_size * num_agents, 23),
            torch.zeros(batch_size * num_agents, 16),
            torch.zeros(batch_size * num_agents, 16),
            actions.reshape(batch_size * num_agents, 1),
            values.reshape(batch_size * num_agents, 1),
            torch.randn(batch_size * num_agents, 1),
            torch.ones(batch_size * num_agents, 1),
            torch.ones(batch_size * num_agents, 1),
            old_log_probs.reshape(batch_size * num_agents, 1),
            torch.randn(batch_size * num_agents, 1),
            torch.ones(batch_size * num_agents, 5),
            adjacency.reshape(batch_size * num_agents, num_agents),
            obs.reshape(batch_size * num_agents, 23),
            torch.zeros(batch_size, 2, 9),
        )
        result = trainer.ppo_update(sample, episode=0, iter_step=0, obs_dim=7)
        for value in result:
            self.assertTrue(torch.isfinite(torch.as_tensor(value)).all())

    def test_shared_mappo_gnn_single_device_path_still_constructs(self):
        args, _ = self.make_policy()
        args.algorithm_name = "mappo_gnn"
        args.truelyDistributed = False
        policy = TransformerPolicy(
            args,
            Box((7,)),
            Box((9,)),
            Discrete(5),
            3,
            torch.device("cpu"),
        )
        with torch.no_grad():
            encoded = policy.transformer.obs_encoder(
                torch.randn(2, 3, 7), self.batched_edges()
            )
        self.assertEqual(encoded.shape, (2, 3, 16))


if __name__ == "__main__":
    unittest.main()
