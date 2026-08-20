"""Coverage for recurrent MAPPO, IPPO, and consensus-IPPO components."""

from types import SimpleNamespace
import unittest

import torch

from mat.algorithms.mat.algorithm.ippo import IPPO
from mat.algorithms.mat.algorithm.transformer_policy import TransformerPolicy
from mat.config import get_config


class Box:
    def __init__(self, size):
        self.shape = (size,)


class Discrete:
    def __init__(self, size):
        self.n = size


class IPPOGRUTest(unittest.TestCase):
    def _model(self, share_policy):
        args = SimpleNamespace(
            share_policy=share_policy,
            use_actor_gru=True,
            use_critic_gru=True,
        )
        return IPPO(
            args, state_dim=7, obs_dim=5, action_dim=4, n_agent=3,
            n_block=1, n_embd=8, n_head=1, num_quants=1,
        )

    def test_actor_and_critic_grus_work_for_shared_and_independent_ippo(self):
        for share_policy in (True, False):
            with self.subTest(share_policy=share_policy):
                model = self._model(share_policy)
                state = torch.randn(6, 3, 7)
                obs = torch.randn(6, 3, 5)
                actions = torch.randint(0, 4, (6, 3, 1))
                initial_state = torch.zeros(2, 3, 8)
                masks = torch.ones(6, 3, 1)

                log_probs, values, entropy = model(
                    state, obs, actions,
                    rnn_states_actor=initial_state,
                    rnn_states_critic=initial_state,
                    masks=masks, sequence_length=3,
                )
                rollout = model.get_actions(
                    state[:2], obs[:2],
                    rnn_states_actor=initial_state,
                    rnn_states_critic=initial_state,
                    masks=masks[:2], return_rnn_states=True,
                )

                self.assertEqual(log_probs.shape, (6, 3, 1))
                self.assertEqual(values.shape, (6, 3, 1))
                self.assertEqual(entropy.shape, (6, 3, 1))
                self.assertEqual(rollout[3].shape, (2, 3, 8))
                self.assertEqual(rollout[4].shape, (2, 3, 8))

    def test_episode_mask_resets_recurrent_state(self):
        model = self._model(share_policy=False)
        obs = torch.randn(1, 3, 5)
        state = torch.randn(1, 3, 7)
        stale_state = torch.randn(1, 3, 8)
        reset_mask = torch.zeros(1, 3, 1)

        reset = model.get_actions(
            state, obs, rnn_states_actor=stale_state,
            rnn_states_critic=stale_state, masks=reset_mask,
            deterministic=True, return_rnn_states=True,
        )
        fresh = model.get_actions(
            state, obs, rnn_states_actor=torch.zeros_like(stale_state),
            rnn_states_critic=torch.zeros_like(stale_state), masks=reset_mask,
            deterministic=True, return_rnn_states=True,
        )

        self.assertTrue(torch.allclose(reset[3], fresh[3]))
        self.assertTrue(torch.allclose(reset[4], fresh[4]))

    def test_policy_wrapper_supports_ippo_and_consensus_ippo(self):
        for algorithm_name in ("ippo", "consensus_ippo"):
            with self.subTest(algorithm_name=algorithm_name):
                args = get_config().parse_args([])
                args.algorithm_name = algorithm_name
                args.use_actor_gru = True
                args.use_critic_gru = True
                args.n_embd = 8
                args.n_block = 1
                args.n_head = 1
                args.n_quants = 1
                args.truelyDistributed = False
                policy = TransformerPolicy(
                    args, Box(5), Box(7), Discrete(4), num_agents=3,
                )
                self.assertFalse(policy.transformer.encoder.share_policy)
                self.assertFalse(policy.transformer.decoder.share_policy)
                self.assertEqual(len(policy.transformer.encoder.head_), 3)
                self.assertEqual(len(policy.transformer.decoder.mlp_), 3)
                values, _, _, actor_states, critic_states = policy.get_actions(
                    torch.randn(2, 3, 7), torch.randn(2, 3, 5),
                    torch.zeros(2, 3, 1, 8),
                    torch.zeros(2, 3, 1, 8), torch.ones(2, 3, 1),
                    deterministic=True,
                )
                self.assertEqual(values.shape, (6, 1))
                self.assertEqual(actor_states.shape, (6, 8))
                self.assertEqual(critic_states.shape, (6, 8))

    def test_ippo_sharing_requires_explicit_opt_in(self):
        args = get_config().parse_args([])
        args.algorithm_name = "ippo"
        args.ippo_share_policy = True
        args.n_embd = 8
        args.n_block = 1
        args.n_head = 1
        args.n_quants = 1
        args.truelyDistributed = False
        policy = TransformerPolicy(args, Box(5), Box(7), Discrete(4), num_agents=3)
        self.assertTrue(policy.transformer.encoder.share_policy)
        self.assertTrue(policy.transformer.decoder.share_policy)
        self.assertFalse(hasattr(policy.transformer.encoder, "head_"))
        self.assertFalse(hasattr(policy.transformer.decoder, "mlp_"))

    def test_mappo_preserves_independent_actor_and_critic_networks(self):
        args = get_config().parse_args([])
        args.algorithm_name = "mappo"
        args.share_policy = False
        args.use_centralized_critic = True
        args.use_actor_gru = True
        args.use_critic_gru = True
        args.n_embd = 8
        args.n_block = 1
        args.n_head = 1
        args.n_quants = 1
        args.truelyDistributed = False

        policy = TransformerPolicy(
            args, Box(5), Box(7), Discrete(4), num_agents=3,
        )

        encoder = policy.transformer.encoder
        decoder = policy.transformer.decoder
        self.assertTrue(encoder.use_centralized_critic)
        self.assertFalse(encoder.share_policy)
        self.assertFalse(decoder.share_policy)
        self.assertEqual(len(encoder.head_), 3)
        self.assertEqual(len(decoder.mlp_), 3)
        self.assertEqual(len(encoder.gru_), 3)
        self.assertEqual(len(decoder.gru_), 3)
        self.assertEqual(len({id(module) for module in encoder.head_}), 3)
        self.assertEqual(len({id(module) for module in decoder.mlp_}), 3)


if __name__ == "__main__":
    unittest.main()
