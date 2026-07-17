import unittest
from types import SimpleNamespace

import torch

from mat.algorithms.mat.algorithm.dg_mat import DGMAT


class DGMATTest(unittest.TestCase):
    def setUp(self):
        args = SimpleNamespace(dg_mat_dropout=0.0, dg_mat_ff_mult=2)
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

    def test_non_neighbor_cannot_change_receiver_context(self):
        first = self.model.actor_attention(self.obs, self.adjacency)
        changed_obs = self.obs.clone()
        changed_obs[:, 2, :] += 100.0
        second = self.model.actor_attention(changed_obs, self.adjacency)

        # Agent 2 is not a one-hop neighbor of receiver 0.
        torch.testing.assert_close(first[:, 0, :], second[:, 0, :])

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
            actor_grads = [
                parameter.grad
                for parameter in self.model.actor_attention.agent_blocks[
                    agent_id
                ].parameters()
            ]
            critic_grads = [
                parameter.grad
                for parameter in self.model.critic_attention.agent_blocks[
                    agent_id
                ].parameters()
            ]
            self.assertTrue(any(gradient is not None for gradient in actor_grads))
            self.assertTrue(any(gradient is not None for gradient in critic_grads))


if __name__ == "__main__":
    unittest.main()
