# DG-MAPPO hop-wise critic consistency

`analyze_hop_critic_consistency.py` evaluates the saved DG-MAPPO critic at
multiple message-passing depths on the **same states**. The trained full-hop
policy generates all actions. Realized discounted return-to-go is the empirical
target, so a MAPPO critic trained under a different policy is not treated as
ground truth.

Run from the repository root on Palmetto after syncing the branch:

```bash
python mat/scripts/analysis/analyze_hop_critic_consistency.py \
  --run_config /home/shahils/Desktop/gitBackupRepo/DG-MAPPO/mat/scripts/results/long_range_predator_prey/LongRangePredatorPreyContinuous-v0/10pred_5prey/mappo_dgnn_dsgd/10v5_easy/wandb/run-20260902_153523-f3r2xidt/files/config.yaml \
  --checkpoint /home/shahils/Desktop/gitBackupRepo/DG-MAPPO/mat/scripts/results/long_range_predator_prey/LongRangePredatorPreyContinuous-v0/10pred_5prey/mappo_dgnn_dsgd/10v5_easy/wandb/run-20260902_153523-f3r2xidt/files/transformer_3124.pt \
  --analysis_episodes 200 \
  --analysis_output_dir /path/to/critic_hop_analysis
```

The command writes:

- `hop_critic_consistency.pdf/png`: held-out error and correlation versus hops;
- `critic_trajectory_episode_*.pdf/png`: one held-out trajectory;
- `hop_metrics.csv`: point estimates and episode-bootstrap 95% intervals;
- `critic_samples.csv`: all paired state-level measurements;
- `affine_calibration.csv`: calibration coefficients; and
- `analysis_summary.json`: configuration and interpretation metadata.

The transformer checkpoint does not contain the training-time ValueNorm state.
The script therefore fits a per-agent, per-hop affine calibration using only a
random half of the episodes, and evaluates MAE/RMSE on the other half. Pearson
and Spearman correlations are computed from raw critic values. Metrics give
each held-out episode equal weight, and confidence intervals use an episode-
cluster bootstrap. This is an
empirical consistency check; it is not a proof that error must decrease
monotonically with message-passing depth.

By default, actions are sampled from the learned policy, matching the policy
whose value function was trained. Add `--deterministic` for a separate
mean-action sensitivity analysis.

## Repeated-rollout initial-state analysis

`analyze_initial_state_value_consistency.py` provides the cleaner estimate of
the value-function target. By default it samples 20 initial states and runs 20
independent stochastic-policy trajectories from each identical initial state.
Every rollout stops on environment termination or after 200 steps. Run it with:

```bash
python mat/scripts/analysis/analyze_initial_state_value_consistency.py \
  --run_config /path/to/wandb/run/files/config.yaml \
  --checkpoint /path/to/wandb/run/files/transformer_3124.pt \
  --full_checkpoint /path/to/checkpoints/seed2/checkpoint_3124.pt \
  --analysis_device cuda \
  --analysis_output_dir /path/to/initial_state_value_analysis
```

The matching seed-2 full checkpoint is accepted so future or repaired
checkpoints can supply ValueNorm. The current checkpoint's
`value_normalizer_state_dict` is empty, however, so the script falls back to a
held-out affine reward-scale map. It fits that map only from the trained
`K=5` critic on calibration initial states and applies the same map to every
hop. Remaining initial states are used for metrics and paired state-cluster
bootstrap intervals.
