#!/usr/bin/env python
"""Measure DG-MAPPO critic consistency as message-passing depth changes.

The trained full-hop policy generates every trajectory.  At each visited state,
the same critic is evaluated counterfactually with K=0,...,K_train message-
passing rounds.  The empirical target is the realized discounted return-to-go,
not a critic from a different policy.

Transformer-only checkpoints do not contain ValueNorm running statistics.  To
avoid treating normalized critic outputs as environment-scale returns, this
script fits a separate affine map for each (hop, agent) on calibration episodes
and reports errors only on disjoint held-out episodes.  Pearson and Spearman
correlations are invariant to the missing affine scale and are also reported.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mat.config import get_config
from mat.scripts.eval.eval_long_range_predator_prey import get_batch_edge_index
from mat.scripts.train.train_long_range_predator_prey import (
    configure_algorithm,
    make_eval_env,
    optional_wandb,
    parse_args,
    seed_everything,
)

optional_wandb(False)

from mat.algorithms.mat.algorithm.transformer_policy import TransformerPolicy


SUPPORTED_ALGORITHMS = {"mappo_dgnn", "mappo_dgnn_dsgd"}


def load_wandb_defaults(path: str | Path) -> dict:
    """Read values from a W&B config.yaml without importing wandb."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency failure on cluster
        raise RuntimeError("PyYAML is required to read the W&B run config.") from exc

    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    defaults = {}
    for key, entry in config.items():
        if key.startswith("_"):
            continue
        defaults[key] = entry.get("value") if isinstance(entry, dict) and "value" in entry else entry
    return defaults


def parse_analysis_args(argv):
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--run_config")
    bootstrap_args, _ = bootstrap.parse_known_args(argv)

    parser = get_config()
    parser.add_argument("--run_config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--analysis_output_dir", default=None)
    parser.add_argument("--analysis_episodes", type=int, default=200)
    parser.add_argument("--analysis_seed", type=int, default=2026)
    parser.add_argument("--analysis_device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--hop_counts", type=int, nargs="+", default=None)
    parser.add_argument("--calibration_fraction", type=float, default=0.5)
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    parser.add_argument("--deterministic", action="store_true", default=False)
    parser.add_argument("--stochastic", action="store_false", dest="deterministic")

    config_defaults = (
        load_wandb_defaults(bootstrap_args.run_config)
        if bootstrap_args.run_config is not None
        else {}
    )
    args = parse_args(argv, parser)

    # parse_args registers the environment-specific flags, so apply W&B values
    # afterward while preserving every option explicitly supplied on this CLI.
    explicit_destinations = set()
    for action in parser._actions:
        for option in action.option_strings:
            if any(token == option or token.startswith(f"{option}=") for token in argv):
                explicit_destinations.add(action.dest)
    known_destinations = {action.dest for action in parser._actions}
    for key, value in config_defaults.items():
        if key in known_destinations and key not in explicit_destinations:
            setattr(args, key, value)
    if (
        "env_episode_length" not in explicit_destinations
        and config_defaults.get("env_episode_length") is None
    ):
        args.env_episode_length = args.episode_length
    configure_algorithm(args)

    args.use_eval = True
    args.use_wandb = False
    args.n_eval_rollout_threads = 1
    args.n_rollout_threads = 1
    if args.env_device.lower().startswith("cuda") and not torch.cuda.is_available():
        args.env_device = "cpu"
    return args


def choose_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--analysis_device cuda was requested, but CUDA is unavailable.")
    use_cuda = torch.cuda.is_available() if name == "auto" else name == "cuda"
    return torch.device("cuda:0" if use_cuda else "cpu")


def discounted_returns(rewards, gamma: float) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=np.float64)
    returns = np.empty_like(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + gamma * running
        returns[index] = running
    return returns


def split_episode_ids(episode_ids, fraction: float, seed: int):
    episode_ids = np.unique(np.asarray(episode_ids, dtype=np.int64))
    if episode_ids.size < 2:
        raise ValueError("At least two episodes are required for calibration and evaluation.")
    if not 0.0 < fraction < 1.0:
        raise ValueError("--calibration_fraction must lie strictly between 0 and 1.")
    shuffled = episode_ids.copy()
    np.random.default_rng(seed).shuffle(shuffled)
    n_calibration = int(round(fraction * shuffled.size))
    n_calibration = min(max(n_calibration, 1), shuffled.size - 1)
    return np.sort(shuffled[:n_calibration]), np.sort(shuffled[n_calibration:])


def fit_affine(x, y):
    """Least-squares y = slope*x + intercept, robust to a constant x."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    centered = x - x.mean()
    denominator = float(np.dot(centered, centered))
    slope = 0.0 if denominator <= np.finfo(float).eps else float(np.dot(centered, y - y.mean()) / denominator)
    return slope, float(y.mean() - slope * x.mean())


def rankdata(values):
    """Average ranks for ties, matching scipy.stats.rankdata(method='average')."""
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def correlation(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or np.std(x) <= np.finfo(float).eps or np.std(y) <= np.finfo(float).eps:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


@contextmanager
def temporary_hops(policy, hops: int):
    encoder = policy.transformer.obs_encoder
    original = int(encoder.K)
    encoder.K = int(hops)
    try:
        yield
    finally:
        encoder.K = original


def make_policy(args, envs, device):
    policy = TransformerPolicy(
        args,
        envs.observation_space[0],
        envs.share_observation_space[0],
        envs.action_space[0],
        envs.n_agents,
        device=device,
    )
    policy.restore(args.checkpoint, allow_partial=False)
    policy.eval()
    return policy


def collect_episode(args, envs, policy, device, episode_id: int, hops):
    obs, share_obs, _ = envs.reset()
    n_agents = envs.n_agents
    actor_state = torch.zeros((1, n_agents, args.n_embd), dtype=torch.float32, device=device)
    critic_state = torch.zeros_like(actor_state)
    masks = torch.ones((1, n_agents, 1), dtype=torch.float32, device=device)
    rewards = []
    state_values = []

    for step in range(args.env_episode_length):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        share_obs_t = torch.as_tensor(share_obs, dtype=torch.float32, device=device)
        edge_index = get_batch_edge_index(envs.get_edge_index_matrix(), n_agents, device)
        values_by_hop = {}
        with torch.no_grad():
            for hop in hops:
                with temporary_hops(policy, hop):
                    value = policy.get_values(
                        share_obs_t,
                        obs_t,
                        critic_state,
                        masks,
                        available_actions=None,
                        batched_edge_index=edge_index,
                    )
                values_by_hop[hop] = value[:, 0].detach().cpu().numpy().copy()

            with temporary_hops(policy, max(hops)):
                _, actions, _, actor_state, critic_state = policy.get_actions(
                    share_obs_t,
                    obs_t,
                    actor_state,
                    critic_state,
                    masks,
                    available_actions=None,
                    batched_edge_index=edge_index,
                    deterministic=args.deterministic,
                )

        actions = actions.reshape(1, n_agents, -1).detach().cpu().numpy()
        actor_state = actor_state.reshape(1, n_agents, -1).detach()
        critic_state = critic_state.reshape(1, n_agents, -1).detach()
        obs, share_obs, reward, dones, _, _ = envs.step(actions)
        rewards.append(float(np.asarray(reward).mean()))
        state_values.append(values_by_hop)
        done = bool(np.all(dones))
        masks.fill_(0.0 if done else 1.0)
        if done:
            break

    returns = discounted_returns(rewards, args.gamma)
    rows = []
    for step, (target, values_by_hop) in enumerate(zip(returns, state_values)):
        for hop in hops:
            for agent, value in enumerate(values_by_hop[hop]):
                rows.append(
                    {
                        "episode": episode_id,
                        "step": step,
                        "agent": agent,
                        "hop": hop,
                        "raw_value": float(value),
                        "return_to_go": float(target),
                        "episode_return": float(np.sum(rewards)),
                    }
                )
    return rows


def _macro_correlations(rows):
    pearsons = []
    spearmans = []
    for agent in sorted({row["agent"] for row in rows}):
        subset = [row for row in rows if row["agent"] == agent]
        x = np.asarray([row["raw_value"] for row in subset])
        y = np.asarray([row["return_to_go"] for row in subset])
        pearsons.append(correlation(x, y))
        spearmans.append(correlation(rankdata(x), rankdata(y)))
    finite_p = [value for value in pearsons if np.isfinite(value)]
    finite_s = [value for value in spearmans if np.isfinite(value)]
    return (
        float(np.mean(finite_p)) if finite_p else float("nan"),
        float(np.mean(finite_s)) if finite_s else float("nan"),
    )


def _metrics(rows):
    errors = np.asarray([row["calibrated_value"] - row["return_to_go"] for row in rows])
    pearson, spearman = _macro_correlations(rows)
    return {
        "mae": float(np.mean(np.abs(errors))),
        "mse": float(np.mean(np.square(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "pearson_macro": pearson,
        "spearman_macro": spearman,
    }


def _aggregate_episode_metrics(episode_metrics):
    def finite_mean(key):
        values = [row[key] for row in episode_metrics if np.isfinite(row[key])]
        return float(np.mean(values)) if values else float("nan")

    mse = finite_mean("mse")
    return {
        "mae": finite_mean("mae"),
        "mse": mse,
        "rmse": float(np.sqrt(mse)) if np.isfinite(mse) else float("nan"),
        "pearson_macro": finite_mean("pearson_macro"),
        "spearman_macro": finite_mean("spearman_macro"),
    }


def calibrate_and_score(rows, fraction: float, seed: int, bootstrap_samples: int):
    calibration_ids, test_ids = split_episode_ids([row["episode"] for row in rows], fraction, seed)
    calibration_set = set(calibration_ids.tolist())
    test_set = set(test_ids.tolist())
    models = {}
    calibration_groups = {}
    for row in rows:
        if row["episode"] in calibration_set:
            calibration_groups.setdefault((row["hop"], row["agent"]), []).append(row)
    for hop in sorted({row["hop"] for row in rows}):
        for agent in sorted({row["agent"] for row in rows}):
            subset = calibration_groups[(hop, agent)]
            models[(hop, agent)] = fit_affine(
                [row["raw_value"] for row in subset],
                [row["return_to_go"] for row in subset],
            )

    for row in rows:
        slope, intercept = models[(row["hop"], row["agent"])]
        row["calibrated_value"] = slope * row["raw_value"] + intercept
        row["split"] = "calibration" if row["episode"] in calibration_set else "test"

    rng = np.random.default_rng(seed + 1)
    metrics = []
    for hop in sorted({row["hop"] for row in rows}):
        test_rows = [row for row in rows if row["episode"] in test_set and row["hop"] == hop]
        by_episode = {int(episode): [] for episode in test_ids}
        for row in test_rows:
            by_episode[row["episode"]].append(row)
        by_episode_metrics = {episode: _metrics(episode_rows) for episode, episode_rows in by_episode.items()}
        point = _aggregate_episode_metrics(list(by_episode_metrics.values()))
        point.update({"hop": hop, "n_samples": len(test_rows), "n_test_episodes": len(test_ids)})
        distributions = {key: [] for key in ("mae", "rmse", "pearson_macro", "spearman_macro")}
        for _ in range(max(bootstrap_samples, 0)):
            sampled_ids = rng.choice(test_ids, size=len(test_ids), replace=True)
            sample_metrics = _aggregate_episode_metrics(
                [by_episode_metrics[int(episode)] for episode in sampled_ids]
            )
            for key in distributions:
                if np.isfinite(sample_metrics[key]):
                    distributions[key].append(sample_metrics[key])
        for key, values in distributions.items():
            if values:
                point[f"{key}_ci_low"], point[f"{key}_ci_high"] = [
                    float(value) for value in np.percentile(values, [2.5, 97.5])
                ]
            else:
                point[f"{key}_ci_low"] = point[f"{key}_ci_high"] = float("nan")
        metrics.append(point)
    calibration_rows = [
        {"hop": hop, "agent": agent, "slope": values[0], "intercept": values[1]}
        for (hop, agent), values in sorted(models.items())
    ]
    return metrics, calibration_rows, calibration_ids, test_ids


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_metrics(output_dir: Path, metrics):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hops = np.asarray([row["hop"] for row in metrics])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for key, label, marker in (("mae", "MAE", "o"), ("rmse", "RMSE", "s")):
        values = np.asarray([row[key] for row in metrics])
        low = np.asarray([row[f"{key}_ci_low"] for row in metrics])
        high = np.asarray([row[f"{key}_ci_high"] for row in metrics])
        axes[0].errorbar(hops, values, yerr=np.vstack([values - low, high - values]), marker=marker, capsize=2, label=label)
    for key, label, marker in (("pearson_macro", "Pearson", "o"), ("spearman_macro", "Spearman", "s")):
        values = np.asarray([row[key] for row in metrics])
        low = np.asarray([row[f"{key}_ci_low"] for row in metrics])
        high = np.asarray([row[f"{key}_ci_high"] for row in metrics])
        axes[1].errorbar(hops, values, yerr=np.vstack([values - low, high - values]), marker=marker, capsize=2, label=label)
    axes[0].set_ylabel("Held-out error")
    axes[1].set_ylabel("Held-out correlation")
    for axis in axes:
        axis.set_xlabel("Message-passing hops $K$")
        axis.set_xticks(hops)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"hop_critic_consistency.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory(output_dir: Path, rows, test_ids, hops):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episode = int(test_ids[0])
    episode_rows = [row for row in rows if row["episode"] == episode]
    steps = sorted({row["step"] for row in episode_rows})
    target = [np.mean([row["return_to_go"] for row in episode_rows if row["step"] == step]) for step in steps]
    fig, axis = plt.subplots(figsize=(7.2, 3.3))
    axis.plot(steps, target, color="black", linewidth=2.0, label="realized return-to-go")
    colors = plt.cm.viridis(np.linspace(0.12, 0.9, len(hops)))
    for color, hop in zip(colors, hops):
        values = [
            np.mean([row["calibrated_value"] for row in episode_rows if row["step"] == step and row["hop"] == hop])
            for step in steps
        ]
        axis.plot(steps, values, color=color, linewidth=1.2, label=f"critic $K={hop}$")
    axis.set_xlabel("Environment step")
    axis.set_ylabel("Discounted return / calibrated value")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"critic_trajectory_episode_{episode}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main(argv):
    args = parse_analysis_args(argv)
    if args.algorithm_name not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"Expected one of {sorted(SUPPORTED_ALGORITHMS)}, got {args.algorithm_name!r}.")
    if args.use_critic_gru:
        raise ValueError("Hop truncation currently requires a feed-forward critic so every K uses the same state history.")
    trained_hops = int(args.iterations)
    hops = sorted(set(args.hop_counts if args.hop_counts is not None else range(trained_hops + 1)))
    if not hops or min(hops) < 0 or max(hops) > trained_hops:
        raise ValueError(f"--hop_counts must be within [0, {trained_hops}].")
    if max(hops) != trained_hops:
        raise ValueError(f"Include trained depth K={trained_hops}; it is used to generate the fixed trajectories.")
    if args.analysis_episodes < 2:
        raise ValueError("--analysis_episodes must be at least 2.")

    # Use an analysis-specific environment seed rather than silently reusing
    # the training seed loaded from W&B.
    args.seed = int(args.analysis_seed)
    seed_everything(args.seed, deterministic=args.cuda_deterministic)
    device = choose_device(args.analysis_device)
    output_dir = Path(args.analysis_output_dir) if args.analysis_output_dir else Path(args.checkpoint).parent / "critic_hop_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    envs = make_eval_env(args)
    rows = []
    try:
        policy = make_policy(args, envs, device)
        for episode in range(args.analysis_episodes):
            episode_rows = collect_episode(args, envs, policy, device, episode, hops)
            rows.extend(episode_rows)
            episode_return = episode_rows[0]["episode_return"]
            print(f"episode={episode + 1}/{args.analysis_episodes} return={episode_return:.3f}", flush=True)
    finally:
        envs.close()

    metrics, calibration, calibration_ids, test_ids = calibrate_and_score(
        rows, args.calibration_fraction, args.analysis_seed, args.bootstrap_samples
    )
    write_csv(output_dir / "critic_samples.csv", rows)
    write_csv(output_dir / "hop_metrics.csv", metrics)
    write_csv(output_dir / "affine_calibration.csv", calibration)
    plot_metrics(output_dir, metrics)
    plot_trajectory(output_dir, rows, test_ids, hops)

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "run_config": str(Path(args.run_config).resolve()),
        "algorithm": args.algorithm_name,
        "trained_hops": trained_hops,
        "evaluated_hops": hops,
        "trajectory_policy_hops": trained_hops,
        "episodes": args.analysis_episodes,
        "analysis_seed": args.analysis_seed,
        "calibration_episode_ids": calibration_ids.tolist(),
        "test_episode_ids": test_ids.tolist(),
        "gamma": args.gamma,
        "deterministic_policy": args.deterministic,
        "empirical_target": "Monte Carlo discounted return-to-go under the trained full-hop DG policy",
        "value_norm_state_available": False,
        "error_scale_note": "Per-(hop,agent) affine maps were fit on calibration episodes only.",
        "interpretation": "Empirical consistency diagnostic; not a proof of Lemma 1 or monotonicity in K.",
        "metrics": metrics,
    }
    with (output_dir / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(summary), handle, indent=2)
    print(f"wrote analysis to {output_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
