#!/usr/bin/env python
"""Compare initial critic values with repeated-rollout Monte Carlo returns.

For each sampled initial state, the environment is reset to exactly the same
state for several independent stochastic-policy rollouts.  The mean discounted
return estimates V^pi(s_0).  The acting policy always uses its trained graph
depth; only the critic evaluation depth is truncated counterfactually.
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mat.scripts.analysis.analyze_hop_critic_consistency import (
    SUPPORTED_ALGORITHMS,
    choose_device,
    correlation,
    fit_affine,
    json_safe,
    make_policy,
    parse_analysis_args,
    rankdata,
    split_episode_ids,
    temporary_hops,
    write_csv,
)
from mat.scripts.eval.eval_long_range_predator_prey import get_batch_edge_index
from mat.scripts.train.train_long_range_predator_prey import make_eval_env, seed_everything
from mat.utils.valuenorm import ValueNorm


def reset_to_initial_seed(envs, seed: int):
    """Reset the single evaluation environment to a reproducible initial state."""
    if not hasattr(envs, "envs") or len(envs.envs) != 1:
        raise ValueError("Initial-state analysis requires one ShareDummyVecEnv environment.")
    obs, share_obs, available_actions = envs.envs[0].reset(seed=int(seed))
    return tuple(np.expand_dims(value, axis=0) for value in (obs, share_obs, available_actions))


def initial_critic_values(policy, envs, obs, share_obs, device, hops):
    n_agents = envs.n_agents
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    share_obs_t = torch.as_tensor(share_obs, dtype=torch.float32, device=device)
    critic_state = torch.zeros((1, n_agents, policy.n_embd), dtype=torch.float32, device=device)
    masks = torch.ones((1, n_agents, 1), dtype=torch.float32, device=device)
    edge_index = get_batch_edge_index(envs.get_edge_index_matrix(), n_agents, device)
    values = {}
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
            values[hop] = value[:, 0].detach().cpu().numpy().copy()
    return values


def rollout_from_initial_state(
    args, envs, policy, device, initial_seed, rollout_seed, trained_hops, expected_initial_obs
):
    # The environment generator is reset to initial_seed, while policy sampling
    # uses a distinct seed. Dynamics after reset are deterministic in this env.
    seed_everything(rollout_seed, deterministic=args.cuda_deterministic)
    obs, share_obs, _ = reset_to_initial_seed(envs, initial_seed)
    if not np.array_equal(obs, expected_initial_obs):
        raise RuntimeError("Repeated rollout did not reproduce the requested initial state exactly.")
    n_agents = envs.n_agents
    actor_state = torch.zeros((1, n_agents, args.n_embd), dtype=torch.float32, device=device)
    critic_state = torch.zeros_like(actor_state)
    masks = torch.ones((1, n_agents, 1), dtype=torch.float32, device=device)
    discounted_return = 0.0
    discount = 1.0

    for step in range(200):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        share_obs_t = torch.as_tensor(share_obs, dtype=torch.float32, device=device)
        edge_index = get_batch_edge_index(envs.get_edge_index_matrix(), n_agents, device)
        with torch.no_grad(), temporary_hops(policy, trained_hops):
            _, actions, _, actor_state, critic_state = policy.get_actions(
                share_obs_t,
                obs_t,
                actor_state,
                critic_state,
                masks,
                available_actions=None,
                batched_edge_index=edge_index,
                deterministic=False,
            )
        actions = actions.reshape(1, n_agents, -1).detach().cpu().numpy()
        actor_state = actor_state.reshape(1, n_agents, -1).detach()
        critic_state = critic_state.reshape(1, n_agents, -1).detach()
        obs, share_obs, rewards, dones, _, _ = envs.step(actions)
        discounted_return += discount * float(np.asarray(rewards).mean())
        discount *= args.gamma
        done = bool(np.all(dones))
        masks.fill_(0.0 if done else 1.0)
        if done:
            return discounted_return, step + 1
    return discounted_return, 200


def load_value_normalizer(path, num_agents, num_quants, device):
    if path is None:
        return None, "no full checkpoint supplied"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("value_normalizer_state_dict")
    if not state:
        return None, "full checkpoint contains an empty ValueNorm state dictionary"
    normalizer = ValueNorm(num_agents, num_quants, norm_axes=0, device=device)
    normalizer.load_state_dict(state)
    normalizer.eval()
    return normalizer, "loaded from full checkpoint"


def apply_reward_scale(rows, trained_hops, normalizer, calibration_fraction, seed):
    state_ids = sorted({row["initial_state"] for row in rows})
    if normalizer is not None:
        for row in rows:
            raw = np.zeros((1, normalizer.running_mean.shape[0], 1), dtype=np.float32)
            raw[0, row["agent"], 0] = row["raw_value"]
            row["scaled_value"] = float(normalizer.denormalize(raw)[0, row["agent"], 0].cpu())
            row["split"] = "test"
        return [], state_ids, [], "checkpoint_valuenorm"

    calibration_ids, test_ids = split_episode_ids(state_ids, calibration_fraction, seed)
    calibration_set = set(calibration_ids.tolist())
    models = {}
    # ValueNorm is shared across depths. Fit it only from the actually trained
    # K_max critic, then apply the identical per-agent map to every hop.
    for agent in sorted({row["agent"] for row in rows}):
        subset = [
            row for row in rows
            if row["initial_state"] in calibration_set
            and row["hop"] == trained_hops
            and row["agent"] == agent
        ]
        models[agent] = fit_affine(
            [row["raw_value"] for row in subset],
            [row["mc_return_mean"] for row in subset],
        )
    for row in rows:
        slope, intercept = models[row["agent"]]
        row["scaled_value"] = slope * row["raw_value"] + intercept
        row["split"] = "calibration" if row["initial_state"] in calibration_set else "test"
    calibration = [
        {"agent": agent, "slope": values[0], "intercept": values[1], "fit_hop": trained_hops}
        for agent, values in sorted(models.items())
    ]
    return calibration_ids.tolist(), test_ids.tolist(), calibration, "held_out_affine_from_trained_hop"


def metrics_for_rows(rows):
    errors = np.asarray([row["scaled_value"] - row["mc_return_mean"] for row in rows])
    pearson = []
    spearman = []
    for agent in sorted({row["agent"] for row in rows}):
        subset = [row for row in rows if row["agent"] == agent]
        raw = [row["raw_value"] for row in subset]
        target = [row["mc_return_mean"] for row in subset]
        pearson.append(correlation(raw, target))
        spearman.append(correlation(rankdata(raw), rankdata(target)))
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "pearson_macro": float(np.nanmean(pearson)),
        "spearman_macro": float(np.nanmean(spearman)),
    }


def score_hops(rows, test_ids, hops, bootstrap_samples, seed):
    test_set = set(test_ids)
    by_hop_state = {
        (hop, state): [row for row in rows if row["hop"] == hop and row["initial_state"] == state]
        for hop in hops for state in test_ids
    }
    point = {}
    for hop in hops:
        point[hop] = metrics_for_rows([row for state in test_ids for row in by_hop_state[(hop, state)]])

    rng = np.random.default_rng(seed + 1)
    draws = {hop: {key: [] for key in point[hop]} for hop in hops}
    delta_draws = {hop: {key: [] for key in point[hop]} for hop in hops}
    for _ in range(max(bootstrap_samples, 0)):
        sampled = rng.choice(test_ids, size=len(test_ids), replace=True)
        sampled_metrics = {}
        for hop in hops:
            sampled_rows = [row for state in sampled for row in by_hop_state[(hop, int(state))]]
            sampled_metrics[hop] = metrics_for_rows(sampled_rows)
            for key, value in sampled_metrics[hop].items():
                draws[hop][key].append(value)
                delta_draws[hop][key].append(value - sampled_metrics[hops[0]][key])

    output = []
    for hop in hops:
        row = {"hop": hop, "n_test_states": len(test_set), **point[hop]}
        for key in point[hop]:
            values = draws[hop][key]
            deltas = delta_draws[hop][key]
            row[f"{key}_ci_low"], row[f"{key}_ci_high"] = (
                [float(x) for x in np.percentile(values, [2.5, 97.5])]
                if values else [float("nan"), float("nan")]
            )
            row[f"{key}_delta_vs_k0"] = point[hop][key] - point[hops[0]][key]
            row[f"{key}_delta_ci_low"], row[f"{key}_delta_ci_high"] = (
                [float(x) for x in np.percentile(deltas, [2.5, 97.5])]
                if deltas else [float("nan"), float("nan")]
            )
        output.append(row)
    return output


def plot_results(output_dir, rows, metrics, test_ids, hops):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hop_values = np.asarray(hops)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for key, label in (("mae", "MAE"), ("rmse", "RMSE")):
        values = np.asarray([row[key] for row in metrics])
        low = np.asarray([row[f"{key}_ci_low"] for row in metrics])
        high = np.asarray([row[f"{key}_ci_high"] for row in metrics])
        axes[0].errorbar(hop_values, values, yerr=np.vstack([values - low, high - values]), marker="o", capsize=2, label=label)
    for key, label in (("pearson_macro", "Pearson"), ("spearman_macro", "Spearman")):
        values = np.asarray([row[key] for row in metrics])
        low = np.asarray([row[f"{key}_ci_low"] for row in metrics])
        high = np.asarray([row[f"{key}_ci_high"] for row in metrics])
        axes[1].errorbar(hop_values, values, yerr=np.vstack([values - low, high - values]), marker="o", capsize=2, label=label)
    axes[0].set_ylabel("Initial-state held-out error")
    axes[1].set_ylabel("Initial-state held-out correlation")
    for axis in axes:
        axis.set_xlabel("Message-passing hops $K$")
        axis.set_xticks(hop_values)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"initial_state_hop_consistency.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 3.4))
    for state in test_ids:
        sample = next(row for row in rows if row["initial_state"] == state)
        axis.errorbar(state, sample["mc_return_mean"], yerr=1.96 * sample["mc_return_se"], fmt="ko", capsize=2)
    for hop in (hops[0], hops[-1]):
        values = []
        for state in test_ids:
            subset = [row["scaled_value"] for row in rows if row["initial_state"] == state and row["hop"] == hop]
            values.append(float(np.mean(subset)))
        axis.plot(test_ids, values, marker="o", label=f"mean critic $K={hop}$")
    axis.set_xlabel("Held-out initial-state index")
    axis.set_ylabel("Discounted return / critic value")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"initial_state_value_comparison.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv):
    args = parse_analysis_args(argv)
    if args.algorithm_name not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"Expected one of {sorted(SUPPORTED_ALGORITHMS)}, got {args.algorithm_name!r}.")
    if args.initial_state_samples < 2 or args.rollouts_per_initial_state < 2:
        raise ValueError("Use at least two initial states and two rollouts per state.")
    args.env_episode_length = 200
    args.episode_length = 200
    args.n_eval_rollout_threads = 1
    args.env_device = "cpu"
    trained_hops = int(args.iterations)
    hops = sorted(set(args.hop_counts if args.hop_counts is not None else range(trained_hops + 1)))
    if not hops or hops[-1] != trained_hops:
        raise ValueError(f"Evaluated hops must include trained depth K={trained_hops}.")

    device = choose_device(args.analysis_device)
    seed_everything(args.analysis_seed, deterministic=args.cuda_deterministic)
    output_dir = Path(args.analysis_output_dir) if args.analysis_output_dir else Path(args.checkpoint).parent / "initial_state_value_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    envs = make_eval_env(args)
    value_rows = []
    rollout_rows = []
    try:
        policy = make_policy(args, envs, device)
        normalizer, normalizer_note = load_value_normalizer(
            args.full_checkpoint, envs.n_agents, args.n_quants, device
        )
        if normalizer is None:
            warnings.warn(f"ValueNorm unavailable: {normalizer_note}; using held-out affine calibration.")
        for state_id in range(args.initial_state_samples):
            initial_seed = args.analysis_seed * 100_000 + state_id
            initial_obs, initial_share_obs, _ = reset_to_initial_seed(envs, initial_seed)
            values = initial_critic_values(policy, envs, initial_obs, initial_share_obs, device, hops)
            returns = []
            for rollout_id in range(args.rollouts_per_initial_state):
                rollout_seed = args.analysis_seed * 1_000_000 + state_id * 10_000 + rollout_id
                value, steps = rollout_from_initial_state(
                    args,
                    envs,
                    policy,
                    device,
                    initial_seed,
                    rollout_seed,
                    trained_hops,
                    initial_obs,
                )
                returns.append(value)
                rollout_rows.append({
                    "initial_state": state_id,
                    "initial_seed": initial_seed,
                    "rollout": rollout_id,
                    "rollout_seed": rollout_seed,
                    "discounted_return": value,
                    "steps": steps,
                })
            mean = float(np.mean(returns))
            std = float(np.std(returns, ddof=1))
            se = std / math.sqrt(len(returns))
            for hop in hops:
                for agent, raw_value in enumerate(values[hop]):
                    value_rows.append({
                        "initial_state": state_id,
                        "initial_seed": initial_seed,
                        "hop": hop,
                        "agent": agent,
                        "raw_value": float(raw_value),
                        "mc_return_mean": mean,
                        "mc_return_std": std,
                        "mc_return_se": se,
                    })
            print(f"state={state_id + 1}/{args.initial_state_samples} MC_return={mean:.3f} SE={se:.3f}", flush=True)
    finally:
        envs.close()

    calibration_ids, test_ids, calibration, scale_method = apply_reward_scale(
        value_rows, trained_hops, normalizer, args.calibration_fraction, args.analysis_seed
    )
    metrics = score_hops(value_rows, test_ids, hops, args.bootstrap_samples, args.analysis_seed)
    write_csv(output_dir / "initial_state_values.csv", value_rows)
    write_csv(output_dir / "rollout_returns.csv", rollout_rows)
    write_csv(output_dir / "initial_state_hop_metrics.csv", metrics)
    write_csv(output_dir / "reward_scale_calibration.csv", calibration)
    plot_results(output_dir, value_rows, metrics, test_ids, hops)
    summary = {
        "initial_state_samples": args.initial_state_samples,
        "rollouts_per_initial_state": args.rollouts_per_initial_state,
        "maximum_episode_length": 200,
        "early_termination_respected": True,
        "gamma": args.gamma,
        "trained_policy_hops": trained_hops,
        "evaluated_critic_hops": hops,
        "calibration_initial_states": calibration_ids,
        "test_initial_states": test_ids,
        "reward_scale_method": scale_method,
        "value_normalizer_note": normalizer_note,
        "metrics": metrics,
    }
    with (output_dir / "initial_state_analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(summary), handle, indent=2)
    print(f"wrote initial-state analysis to {output_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
