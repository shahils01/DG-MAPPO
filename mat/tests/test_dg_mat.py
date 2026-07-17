import unittest
from types import SimpleNamespace

import torch

from mat.algorithms.mat.algorithm.dg_mat import DGMAT
from mat.algorithms.mat.algorithm.transformer_policy import TransformerPolicy
from mat.utils.shared_buffer import SharedReplayBuffer


class Box:
    def __init__(self, shape):
        self.shape = shape


class Discrete:
    def __init__(self, n):
        self.n = n


class DGMATTest(unittest.TestCase):
    def setUp(self):
        args = SimpleNamespace(
            dg_mat_dropout=0.0,
            dg_mat_ff_mult=2,
            dg_mat_obs_tokens=4,
        )
        self.model = DGMAT(
            args=args,
            state_dim=9,
            obs_dim=7,
            action_dim=5,
            n_agent=3,
            n_block=1,
            n_embd=16,
            n_head=2,
            device=torch.device("cpu"),
            action_type="Discrete",
            num_quants=1,
        )
        self.model.eval()
        self.obs = torch.randn(4, 3, 7)
        self.state = torch.randn(4, 3, 9)
        self.adjacency = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        ).unsqueeze(0).expand(4, -1, -1)

    def test_rollout_and_ppo_log_probs_match(self):
        actions, rollout_log_probs, rollout_values = self.model.get_actions(
            self.state,
            self.obs,
            deterministic=True,
            adjacency_matrix=self.adjacency,
        )
        ppo_log_probs, ppo_values, entropy = self.model(
            self.state,
            self.obs,
            actions,
            adjacency_matrix=self.adjacency,
        )

        self.assertEqual(actions.shape, (4, 3, 1))
        self.assertEqual(rollout_values.shape, (4, 3, 1))
        self.assertEqual(entropy.shape, (4, 3, 1))
        torch.testing.assert_close(rollout_log_probs, ppo_log_probs)
        torch.testing.assert_close(rollout_values, ppo_values)

    def test_local_encoder_creates_sender_owned_messages(self):
        first = self.model.actor_local_encoder(self.obs)
        changed_obs = self.obs.clone()
        changed_obs[:, 2, :] += 100.0
        second = self.model.actor_local_encoder(changed_obs)

        # Changing sender 2's observation changes only sender 2's message.
        torch.testing.assert_close(first[:, 0, :], second[:, 0, :])
        torch.testing.assert_close(first[:, 1, :], second[:, 1, :])
        self.assertFalse(torch.allclose(first[:, 2, :], second[:, 2, :]))

    def test_only_encoded_neighbor_messages_cross_the_graph(self):
        first_messages = self.model.actor_local_encoder(self.obs)
        first_context = self.model.actor_communication(
            first_messages, self.adjacency
        )

        changed_obs = self.obs.clone()
        changed_obs[:, 2, :] += 100.0
        second_messages = self.model.actor_local_encoder(changed_obs)
        second_context = self.model.actor_communication(
            second_messages, self.adjacency
        )

        # Agent 2 is not a one-hop neighbor of receiver 0.
        torch.testing.assert_close(first_context[:, 0, :], second_context[:, 0, :])
        # Agent 2 is a neighbor of receiver 1, so its encoded message matters.
        self.assertFalse(
            torch.allclose(first_context[:, 1, :], second_context[:, 1, :])
        )

    def test_actor_and_critic_attention_receive_gradients(self):
        self.model.train()
        actions = torch.zeros(4, 3, 1, dtype=torch.long)
        log_probs, values, entropy = self.model(
            self.state,
            self.obs,
            actions,
            adjacency_matrix=self.adjacency,
        )
        loss = -(log_probs.mean() + 0.01 * entropy.mean()) + values.square().mean()
        loss.backward()

        for agent_id in range(3):
            actor_encoder_grads = [
                parameter.grad
                for parameter in self.model.actor_local_encoder.agent_encoders[
                    agent_id
                ].parameters()
            ]
            critic_encoder_grads = [
                parameter.grad
                for parameter in self.model.critic_local_encoder.agent_encoders[
                    agent_id
                ].parameters()
            ]
            actor_communication_grads = [
                parameter.grad
                for parameter in self.model.actor_communication.agent_blocks[
                    agent_id
                ].parameters()
            ]
            critic_communication_grads = [
                parameter.grad
                for parameter in self.model.critic_communication.agent_blocks[
                    agent_id
                ].parameters()
            ]
            self.assertTrue(
                any(gradient is not None for gradient in actor_encoder_grads)
            )
            self.assertTrue(
                any(gradient is not None for gradient in critic_encoder_grads)
            )
            self.assertTrue(
                any(gradient is not None for gradient in actor_communication_grads)
            )
            self.assertTrue(
                any(gradient is not None for gradient in critic_communication_grads)
            )

    def test_per_agent_optimizers_own_all_and_only_their_modules(self):
        args = SimpleNamespace(
            algorithm_name="dg_mat",
            lr=5e-4,
            opti_eps=1e-5,
            weight_decay=0.0,
            use_policy_active_masks=True,
            n_quants=1,
            n_embd=16,
            truelyDistributed=True,
            clone_extra_agents_from=None,
            n_block=1,
            n_head=2,
            encode_state=False,
            dec_actor=False,
            share_actor=False,
            env_name="StarCraft2",
            dg_mat_dropout=0.0,
            dg_mat_ff_mult=2,
            dg_mat_obs_tokens=4,
        )
        policy = TransformerPolicy(
            args=args,
            obs_space=Box((7,)),
            cent_obs_space=Box((9,)),
            act_space=Discrete(5),
            num_agents=3,
            device=torch.device("cpu"),
        )

        parameter_sets = [
            {id(parameter) for parameter in policy.agent_parameters(agent_id)}
            for agent_id in range(3)
        ]
        for left in range(3):
            for right in range(left + 1, 3):
                self.assertTrue(parameter_sets[left].isdisjoint(parameter_sets[right]))

        owned_parameters = set().union(*parameter_sets)
        model_parameters = {id(parameter) for parameter in policy.transformer.parameters()}
        self.assertEqual(owned_parameters, model_parameters)

    def test_dg_mat_auto_buffer_storage_is_cpu(self):
        args = SimpleNamespace(
            episode_length=2,
            n_rollout_threads=2,
            n_embd=16,
            recurrent_N=1,
            gamma=0.99,
            gae_lambda=0.95,
            use_gae=True,
            use_popart=False,
            use_valuenorm=False,
            use_proper_time_limits=False,
            algorithm_name="dg_mat",
            n_quants=1,
            buffer_device="auto",
            disable_buffer_pin_memory=False,
        )
        buffer = SharedReplayBuffer(
            args=args,
            num_agents=3,
            obs_space=Box((7,)),
            cent_obs_space=Box((9,)),
            act_space=Discrete(5),
            env_name="StarCraft2",
        )

        self.assertEqual(buffer.storage_device, torch.device("cpu"))
        for value in vars(buffer).values():
            if isinstance(value, torch.Tensor):
                self.assertEqual(value.device, torch.device("cpu"))

        for _ in range(2):
            buffer.insert(
                share_obs=torch.randn(2, 3, 9),
                obs=torch.randn(2, 3, 7),
                rnn_states_actor=torch.zeros(2, 3, 1, 16),
                rnn_states_critic=torch.zeros(2, 3, 1, 16),
                actions=torch.zeros(2, 3, 1),
                action_log_probs=torch.zeros(2, 3, 1),
                value_preds=torch.zeros(2, 3, 1),
                rewards=torch.ones(2, 3, 1),
                masks=torch.ones(2, 3, 1),
                bad_masks=torch.ones(2, 3, 1),
                active_masks=torch.ones(2, 3, 1),
                available_actions=torch.ones(2, 3, 5),
                edge_index=torch.zeros(2, 2, 9),
                adjcency_matrix=torch.eye(3).unsqueeze(0).expand(2, -1, -1),
            )

        buffer.compute_returns(torch.zeros(2, 3, 1))
        sample = next(
            buffer.feed_forward_generator_transformer(
                buffer.advantages,
                mini_batch_size=1,
            )
        )
        self.assertTrue(all(
            value is None or not isinstance(value, torch.Tensor) or value.device.type == "cpu"
            for value in sample
        ))


if __name__ == "__main__":
    unittest.main()
