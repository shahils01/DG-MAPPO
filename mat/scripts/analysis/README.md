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
