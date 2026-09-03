from types import SimpleNamespace

import numpy as np

from mat.scripts.analysis.analyze_hop_critic_consistency import (
    calibrate_and_score,
    discounted_returns,
    fit_affine,
    parse_analysis_args,
    rankdata,
    split_episode_ids,
    temporary_hops,
)


def test_discounted_returns():
    np.testing.assert_allclose(discounted_returns([1.0, 2.0, 3.0], 0.5), [2.75, 3.5, 3.0])


def test_episode_split_is_disjoint_and_exhaustive():
    calibration, test = split_episode_ids([0, 0, 1, 2, 3, 4], 0.6, seed=7)
    assert set(calibration).isdisjoint(test)
    assert set(calibration) | set(test) == {0, 1, 2, 3, 4}


def test_fit_affine_recovers_value_scale():
    x = np.asarray([-2.0, 0.0, 1.0, 4.0])
    slope, intercept = fit_affine(x, 3.0 * x - 5.0)
    assert np.isclose(slope, 3.0)
    assert np.isclose(intercept, -5.0)


def test_rankdata_uses_average_tie_ranks():
    np.testing.assert_allclose(rankdata([30.0, 10.0, 10.0, 20.0]), [4.0, 1.5, 1.5, 3.0])


def test_temporary_hops_restores_encoder_depth():
    policy = SimpleNamespace(transformer=SimpleNamespace(obs_encoder=SimpleNamespace(K=5)))
    with temporary_hops(policy, 2):
        assert policy.transformer.obs_encoder.K == 2
    assert policy.transformer.obs_encoder.K == 5


def test_calibration_is_fit_on_episode_disjoint_split():
    rows = []
    for episode in range(6):
        for step in range(3):
            target = float(episode * 3 + step)
            for agent in range(2):
                for hop in range(2):
                    scale = 2.0 + agent + hop
                    offset = -4.0 + agent
                    rows.append(
                        {
                            "episode": episode,
                            "step": step,
                            "agent": agent,
                            "hop": hop,
                            "raw_value": (target - offset) / scale,
                            "return_to_go": target,
                        }
                    )

    metrics, calibration, calibration_ids, test_ids = calibrate_and_score(
        rows, fraction=0.5, seed=11, bootstrap_samples=10
    )
    assert set(calibration_ids).isdisjoint(test_ids)
    assert len(calibration) == 4
    assert all(metric["mae"] < 1e-10 for metric in metrics)
    assert all(row["split"] in {"calibration", "test"} for row in rows)


def test_wandb_config_reconstructs_env_and_cli_can_override(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "algorithm_name:\n  value: mappo_dgnn_dsgd\n"
        "iterations:\n  value: 5\n"
        "num_predators:\n  value: 10\n"
        "num_prey:\n  value: 5\n",
        encoding="utf-8",
    )
    base = ["--run_config", str(config), "--checkpoint", "unused.pt"]
    args = parse_analysis_args(base)
    assert (args.num_predators, args.num_prey, args.iterations) == (10, 5, 5)
    overridden = parse_analysis_args(base + ["--num_predators", "3"])
    assert overridden.num_predators == 3
