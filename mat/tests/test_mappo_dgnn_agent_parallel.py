import sys
import types
import unittest

import torch
import torch.nn as nn

from mat.algorithms.mat.algorithm.transformer_policy import TransformerPolicy
from mat.algorithms.mat.mat_trainer import MATTrainer
from mat.config import get_config
from mat.utils.shared_buffer import SharedReplayBuffer


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

    def make_policy(self, actor_gru=False, critic_gru=False):
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
            "use_actor_gru": actor_gru,
            "use_critic_gru": critic_gru,
            "use_recurrent_policy": actor_gru or critic_gru,
            "recurrent_N": 1,
            "data_chunk_length": 2,
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
        obs = torch.randn(batch_size, num_agents, 7)
        state = torch.randn(batch_size, num_agents, 9)
        adjacency = torch.ones(batch_size, num_agents, num_agents)
        edge_index = torch.full((batch_size, 2, 9), -1.0)
        self_edges = torch.arange(num_agents, dtype=torch.float32)
        edge_index[:, 0, :num_agents] = self_edges
        edge_index[:, 1, :num_agents] = self_edges
        with torch.no_grad():
            actions, old_log_probs, values = policy.transformer.get_actions(
                state, obs, batched_edge_index=edge_index
            )

        sample = (
            state.reshape(batch_size * num_agents, 9),
            obs.reshape(batch_size * num_agents, 7),
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
            obs.reshape(batch_size * num_agents, 7),
            edge_index,
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

    def test_actor_and_critic_grus_are_independently_selectable(self):
        for actor_gru, critic_gru in ((True, False), (False, True), (True, True)):
            with self.subTest(actor_gru=actor_gru, critic_gru=critic_gru):
                _, policy = self.make_policy(actor_gru, critic_gru)
                model = policy.transformer
                obs = torch.randn(2, 3, 7)
                state = torch.randn(2, 3, 9)
                hidden = torch.zeros(2, 3, 16)
                masks = torch.ones(2, 3, 1)
                result = model.get_actions(
                    state,
                    obs,
                    deterministic=True,
                    batched_edge_index=self.batched_edges(),
                    rnn_states_actor=hidden,
                    rnn_states_critic=hidden,
                    masks=masks,
                    return_rnn_states=True,
                )
                _, _, values, actor_states, critic_states = result

                self.assertEqual(values.shape, (2, 3, 1))
                self.assertEqual(actor_states.shape, (2, 3, 16))
                self.assertEqual(critic_states.shape, (2, 3, 16))
                self.assertEqual(bool(model.decoder.gru_), actor_gru)
                self.assertEqual(bool(model.encoder.gru_), critic_gru)
                if actor_gru:
                    self.assertFalse(torch.equal(actor_states, hidden))
                else:
                    torch.testing.assert_close(actor_states, hidden)
                if critic_gru:
                    self.assertFalse(torch.equal(critic_states, hidden))
                else:
                    torch.testing.assert_close(critic_states, hidden)

    def test_policy_rollout_wrapper_accepts_raw_local_observations(self):
        _, policy = self.make_policy(actor_gru=True, critic_gru=True)
        batch_size = 2
        values, actions, log_probs, actor_states, critic_states = (
            policy.get_actions(
                torch.randn(batch_size, 3, 9),
                torch.randn(batch_size, 3, 7),
                torch.zeros(batch_size, 3, 1, 16),
                torch.zeros(batch_size, 3, 1, 16),
                torch.ones(batch_size, 3, 1),
                torch.ones(batch_size, 3, 5),
                self.batched_edges(),
                deterministic=True,
            )
        )
        self.assertEqual(values.shape, (batch_size * 3, 1))
        self.assertEqual(actions.shape, (batch_size * 3, 1))
        self.assertEqual(log_probs.shape, (batch_size * 3, 1))
        self.assertEqual(actor_states.shape, (batch_size * 3, 16))
        self.assertEqual(critic_states.shape, (batch_size * 3, 16))

    def test_recurrent_state_resets_on_episode_mask(self):
        _, policy = self.make_policy(actor_gru=True, critic_gru=True)
        model = policy.transformer
        model.eval()
        state = torch.randn(2, 3, 9)
        first_obs = torch.randn(2, 3, 7)
        second_obs = torch.randn(2, 3, 7)
        zeros = torch.zeros(2, 3, 16)
        ones = torch.ones(2, 3, 1)

        first = model.get_actions(
            state,
            first_obs,
            deterministic=True,
            batched_edge_index=self.batched_edges(),
            rnn_states_actor=zeros,
            rnn_states_critic=zeros,
            masks=ones,
            return_rnn_states=True,
        )
        history_actor, history_critic = first[3], first[4]
        reset = model.get_actions(
            state,
            second_obs,
            deterministic=True,
            batched_edge_index=self.batched_edges(),
            rnn_states_actor=history_actor,
            rnn_states_critic=history_critic,
            masks=torch.zeros_like(ones),
            return_rnn_states=True,
        )
        fresh = model.get_actions(
            state,
            second_obs,
            deterministic=True,
            batched_edge_index=self.batched_edges(),
            rnn_states_actor=zeros,
            rnn_states_critic=zeros,
            masks=ones,
            return_rnn_states=True,
        )
        for reset_value, fresh_value in zip(reset, fresh):
            torch.testing.assert_close(reset_value, fresh_value)

    def test_critic_is_state_independent_and_gnn_gru_receive_gradients(self):
        _, policy = self.make_policy(actor_gru=True, critic_gru=True)
        model = policy.transformer
        model.eval()
        obs = torch.randn(2, 3, 7)
        state = torch.randn(2, 3, 9)
        hidden = torch.zeros(2, 3, 16)
        masks = torch.ones(2, 3, 1)
        actions = torch.zeros(2, 3, 1, dtype=torch.long)

        log_probs, values, entropy = model(
            state,
            obs,
            actions,
            rnn_states_actor=hidden,
            rnn_states_critic=hidden,
            masks=masks,
            edge_index=self.batched_edges(),
        )
        with torch.no_grad():
            _, changed_values, _ = model(
                state + 1000.0,
                obs,
                actions,
                rnn_states_actor=hidden,
                rnn_states_critic=hidden,
                masks=masks,
                edge_index=self.batched_edges(),
            )
        torch.testing.assert_close(values.detach(), changed_values)

        loss = -log_probs.mean() - 0.01 * entropy.mean() + values.square().mean()
        loss.backward()
        for agent_id in range(3):
            self.assertTrue(
                any(
                    parameter.grad is not None
                    for parameter in model.obs_encoder.agent_encoders[
                        agent_id
                    ].parameters()
                )
            )
            self.assertTrue(
                any(
                    parameter.grad is not None
                    for parameter in model.decoder.gru_[agent_id].parameters()
                )
            )
            self.assertTrue(
                any(
                    parameter.grad is not None
                    for parameter in model.encoder.gru_[agent_id].parameters()
                )
            )

    def test_recurrent_buffer_and_ppo_update_preserve_sequences(self):
        args, policy = self.make_policy(actor_gru=True, critic_gru=True)
        args.episode_length = 4
        args.n_rollout_threads = 2
        args.buffer_device = "cpu"
        args.disable_buffer_pin_memory = True
        args.num_mini_batch = 1
        args.mini_batch_size = 8
        args.ppo_epoch = 1
        buffer = SharedReplayBuffer(
            args,
            3,
            Box((7,)),
            Box((9,)),
            Discrete(5),
            "smacv2",
            training_device=torch.device("cpu"),
        )
        buffer.obs.normal_()
        buffer.share_obs.normal_()
        buffer.actions.zero_()
        buffer.action_log_probs.zero_()
        buffer.value_preds.zero_()
        buffer.returns.normal_()
        buffer.advantages.normal_()
        buffer.adjcency_matrix.fill_(1.0)
        buffer.edge_index.fill_(-1.0)
        self_edges = torch.arange(3, dtype=torch.float32)
        buffer.edge_index[:, :, 0, :3] = self_edges
        buffer.edge_index[:, :, 1, :3] = self_edges

        generator = buffer.recurrent_generator_transformer(
            buffer.advantages, num_mini_batch=1, data_chunk_length=2
        )
        sample = next(generator)
        self.assertEqual(sample[-1], 2)
        # Four chunks (two per environment), each with two ordered timesteps.
        self.assertEqual(sample[1].shape, (4 * 2 * 3, 7))
        self.assertEqual(sample[14].shape, (4 * 2, 2, 9))

        trainer = MATTrainer(args, policy, 3, torch.device("cpu"))
        result = trainer.ppo_update(
            sample, episode=0, iter_step=0, obs_dim=7
        )
        for value in result:
            self.assertTrue(torch.isfinite(torch.as_tensor(value)).all())


if __name__ == "__main__":
    unittest.main()
