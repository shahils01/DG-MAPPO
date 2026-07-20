import os
from types import SimpleNamespace
import unittest

import numpy as np

# PySC2 installations used with SMACv2 may contain older generated stubs.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from mat.envs.smacv2_adapter import SMACv2EnvAdapter, load_smacv2_env_args


class _Unit:
    def __init__(self, x, y, health=1):
        self.pos = SimpleNamespace(x=x, y=y)
        self.health = health


class _FakeSMACv2:
    def __init__(self):
        self.n_agents = 3
        self.n_enemies = 2
        self.agents = {
            0: _Unit(0.0, 0.0),
            1: _Unit(1.0, 0.0),
            2: _Unit(5.0, 0.0),
        }
        self._seed = None
        self._obs = np.arange(12, dtype=np.float32).reshape(3, 4)
        self._state = np.arange(6, dtype=np.float32)
        self.closed = False

    def get_env_info(self):
        return {
            "n_agents": 3,
            "n_actions": 8,
            "obs_shape": 4,
            "state_shape": 6,
            "episode_limit": 100,
        }

    def reset(self):
        return self._obs.copy(), self._state.copy()

    def step(self, actions):
        assert len(actions) == self.n_agents
        self.agents[1].health = 0
        return 2.5, False, {"battle_won": False}

    def get_obs(self):
        return self._obs.copy()

    def get_state(self):
        return self._state.copy()

    def get_avail_actions(self):
        return np.ones((self.n_agents, 8), dtype=np.int64)

    def get_unit_by_id(self, agent_id):
        return self.agents[agent_id]

    def unit_sight_range(self, agent_id):
        del agent_id
        return 2.0

    def get_stats(self):
        return {"battles_won": 1, "battles_game": 4, "timeouts": 2}

    def close(self):
        self.closed = True


def _args(**overrides):
    values = {
        "use_stacked_frames": False,
        "stacked_frames": 1,
        "smacv2_comm_range": None,
        "smacv2_force_connected_graph": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class SMACv2ConfigTest(unittest.TestCase):
    def test_epo_is_the_default_and_overrides_agent_counts(self):
        env_args, scenario, config_path = load_smacv2_env_args(
            "terran_epo", n_units=6, n_enemies=5
        )

        self.assertTrue(config_path.name.endswith("terran_epo.yaml"))
        self.assertEqual(env_args["prob_obs_enemy"], 0.0)
        self.assertFalse(env_args["action_mask"])
        self.assertEqual(env_args["capability_config"]["n_units"], 6)
        self.assertEqual(env_args["capability_config"]["n_enemies"], 5)
        self.assertEqual(scenario, "terran_6_vs_5_epo")

    def test_epo_parameters_can_be_swept(self):
        env_args, scenario, _ = load_smacv2_env_args(
            "protoss_epo", prob_obs_enemy=0.5, action_mask=True
        )

        self.assertEqual(env_args["prob_obs_enemy"], 0.5)
        self.assertTrue(env_args["action_mask"])
        self.assertEqual(scenario, "protoss_5_vs_5_epo")


class SMACv2AdapterTest(unittest.TestCase):
    def test_reset_and_step_match_the_runner_interface(self):
        upstream = _FakeSMACv2()
        env = SMACv2EnvAdapter(_args(), env=upstream, seed=7)

        observations, states, available_actions = env.reset()
        self.assertEqual(observations.shape, (3, 4))
        self.assertEqual(states.shape, (3, 6))
        np.testing.assert_array_equal(states[0], states[2])
        self.assertEqual(available_actions.shape, (3, 8))
        self.assertEqual(env.observation_space[0], [4])
        self.assertEqual(env.share_observation_space[0], [6])
        self.assertEqual(env.action_space[0].n, 8)

        transition = env.step(np.array([[1], [2], [3]]))
        self.assertEqual(len(transition), 6)
        _, _, rewards, dones, infos, next_available_actions = transition
        np.testing.assert_array_equal(rewards, np.full((3, 1), 2.5))
        np.testing.assert_array_equal(dones, [False, True, False])
        self.assertEqual(infos[0]["battles_won"], 1)
        self.assertEqual(infos[0]["battles_game"], 4)
        self.assertFalse(infos[0]["bad_transition"])
        self.assertFalse(infos[0]["won"])
        self.assertEqual(next_available_actions.shape, (3, 8))

    def test_graph_uses_physical_range_and_keeps_disconnections(self):
        env = SMACv2EnvAdapter(_args(), env=_FakeSMACv2())
        adjacency = env.get_agent_communication_matrix()

        expected = np.array(
            [[False, True, False], [True, False, False], [False, False, False]]
        )
        np.testing.assert_array_equal(adjacency, expected)

        edge_index = env.get_edge_index_matrix()
        self.assertEqual(edge_index.shape, (2, 9))
        np.testing.assert_array_equal(edge_index[:, :3], [[0, 0, 0], [0, 1, -1]])
        np.testing.assert_array_equal(edge_index[:, 6:9], [[2, 2, 2], [2, -1, -1]])

        visibility = env.get_visibility_matrix()
        self.assertEqual(visibility.shape, (3, 5))
        np.testing.assert_array_equal(visibility[:, :3], expected)
        self.assertFalse(visibility[:, 3:].any())

    def test_connected_graph_repair_is_opt_in(self):
        env = SMACv2EnvAdapter(
            _args(smacv2_force_connected_graph=True), env=_FakeSMACv2()
        )
        adjacency = env.get_agent_communication_matrix()

        self.assertTrue(adjacency[0, 1])
        self.assertTrue(adjacency[1, 2])
        self.assertEqual(len(env._components(adjacency, [0, 1, 2])), 1)

    def test_stacked_frames_update_declared_shapes(self):
        env = SMACv2EnvAdapter(
            _args(use_stacked_frames=True, stacked_frames=3), env=_FakeSMACv2()
        )

        observations, states, _ = env.reset()
        self.assertEqual(observations.shape, (3, 12))
        self.assertEqual(states.shape, (3, 18))
        self.assertEqual(env.observation_space[0], [12])
        self.assertEqual(env.share_observation_space[0], [18])


if __name__ == "__main__":
    unittest.main()
