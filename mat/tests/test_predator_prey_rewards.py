import torch

from mat.envs.long_range_predator_prey.continuous import (
    LongRangePredatorPreyConfig,
    LongRangePredatorPreyTorchCore,
)


def test_global_reward_remains_the_default_and_is_shared():
    core = LongRangePredatorPreyTorchCore(
        LongRangePredatorPreyConfig(num_envs=2, num_predators=4, num_prey=2, seed=3)
    )
    actions = torch.zeros(2, 4, 2)

    _, _, rewards, _, infos, _ = core.step(actions)

    assert core.cfg.reward_mode == "global"
    torch.testing.assert_close(rewards, rewards[:, :1].expand_as(rewards))
    for env_i in range(2):
        assert rewards[env_i, 0, 0].item() == infos[env_i][0]["team_reward"]


def test_local_reward_average_exactly_recovers_team_reward():
    core = LongRangePredatorPreyTorchCore(
        LongRangePredatorPreyConfig(
            num_envs=2,
            num_predators=4,
            num_prey=2,
            reward_mode="local",
            seed=5,
        )
    )
    actions = torch.tensor(
        [
            [[0.2, -0.1], [0.0, 0.3], [-0.4, 0.2], [0.1, 0.0]],
            [[-0.2, 0.4], [0.3, -0.1], [0.0, 0.2], [-0.3, -0.2]],
        ],
        dtype=torch.float32,
    )

    _, _, rewards, _, infos, _ = core.step(actions)
    local_mean = rewards.squeeze(-1).mean(dim=1)
    team_reward = torch.tensor([env[0]["team_reward"] for env in infos])

    torch.testing.assert_close(local_mean, team_reward)


def test_local_reward_conserves_capture_and_completion_bonuses():
    core = LongRangePredatorPreyTorchCore(
        LongRangePredatorPreyConfig(
            num_envs=1,
            num_predators=2,
            num_prey=1,
            capture_k=1,
            capture_radius=0.35,
            prey_speed_ratio=0.0,
            reward_mode="local",
            seed=6,
        )
    )
    core.predator_pose[0, :, :2] = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    core.prey_pose[0, 0, :2] = torch.tensor([0.1, 0.0])
    core.prev_min_dist[0, 0] = 0.1

    _, _, rewards, dones, infos, _ = core.step(torch.full((1, 2, 2), -1.0))

    assert dones.all()
    assert infos[0][0]["new_captures"] == 1
    assert rewards[0].max() > 40.0  # N times the allocated 10 + 15 event rewards.
    torch.testing.assert_close(
        rewards[0, :, 0].mean(),
        torch.tensor(infos[0][0]["team_reward"]),
    )


def test_metropolis_reward_consensus_preserves_average_and_reduces_disagreement():
    core = LongRangePredatorPreyTorchCore(
        LongRangePredatorPreyConfig(
            num_envs=1,
            num_predators=3,
            num_prey=1,
            comm_radius=1.1,
            ensure_connected_comm_graph=False,
            seed=7,
        )
    )
    core.predator_pose[0, :, :2] = torch.tensor([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    initial = torch.tensor([[3.0, -2.0, 8.0]])
    target = initial.mean(dim=1, keepdim=True)

    aggregated = core._consensus_rewards(initial, steps=30)

    torch.testing.assert_close(aggregated.mean(dim=1), initial.mean(dim=1))
    assert (aggregated - target).abs().max() < (initial - target).abs().max()
    torch.testing.assert_close(aggregated, target.expand_as(aggregated), atol=1e-4, rtol=0.0)


def test_consensus_does_not_change_an_already_shared_reward():
    core = LongRangePredatorPreyTorchCore(
        LongRangePredatorPreyConfig(num_envs=1, num_predators=5, num_prey=1, seed=11)
    )
    shared = torch.full((1, 5), 4.25)
    torch.testing.assert_close(core._consensus_rewards(shared, steps=4), shared)
