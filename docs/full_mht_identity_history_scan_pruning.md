# FullMHT Identity-History Scan-Pruning Probe, 2026-06-26

This probe tests whether scan-time history pressure adds real value to the combined identity-history FullMHT row.

The existing central row is kept unchanged:

```text
FullMHTIdentityHistory
```

The new add-on rows use the same calibrated association, prior-survival, no-prior continuation, and growth-history prediction terms, and add only:

```text
scan_motion_history_weight
```

The frozen weights are:

```text
0.25, 0.50, 1.00
```

Each scan-pruning row has a matching greedy beam-width-1 ablation with the same scan-history weight. This is essential: the add-on is promotable only if the full beam beats the matching greedy history, not merely if another scalar score happens to improve a benchmark table.

## Frozen Artifact

```text
benchmarks/full_mht_identity_history_scan_pruning_manifest.json
```

Rows:

| row | purpose |
| --- | --- |
| `Track2p` | original proposal baseline |
| `FullMHTPrior2` | prior-only FullMHT control |
| `FullMHTIdentityHistory` | central combined identity-history baseline |
| `FullMHTGreedyIdentityHistory` | central greedy control |
| `IdentityHistoryScanPruning025` | full beam with scan-history weight `0.25` |
| `GreedyIdentityHistoryScanPruning025` | matching greedy row |
| `IdentityHistoryScanPruning050` | full beam with scan-history weight `0.50` |
| `GreedyIdentityHistoryScanPruning050` | matching greedy row |
| `IdentityHistoryScanPruning100` | full beam with scan-history weight `1.00` |
| `GreedyIdentityHistoryScanPruning100` | matching greedy row |

## Decision Helper

```text
python -m bayescatrack.experiments.full_mht_identity_history_scan_pruning_decision
```

The helper requires, for every candidate weight:

- no regression against the matching greedy row on pairwise or complete-track micro/macro F1;
- no regression against `FullMHTIdentityHistory` on pairwise or complete-track micro/macro F1;
- complete-track F1 micro advantage over the matching greedy row.

Promotion-level evidence requires at least two nearby weights with complete-track advantage and no regressing neighbor. A single winning weight is treated as exploratory.

## Server Command

```bash
REPO=/home/florianpfaff/codex-runs/BayesCaTrack
PY="$REPO/.venv312/bin/python"
cd "$REPO"
git fetch origin
git checkout codex/full-mht-prototype
git reset --hard origin/codex/full-mht-prototype
export PYTHONPATH="$REPO/src"

"$PY" -m pytest -q \
  tests/test_full_mht_identity_history_scan_pruning_manifest.py \
  tests/test_full_mht_identity_history_scan_pruning_decision.py \
  tests/test_full_mht_scan_history_conflict_demo.py \
  tests/test_full_mht_scan_history_dynamics_integration.py \
  tests/test_full_mht_exposure_audit.py \
  tests/test_full_mht_no_gt_leakage.py

OUT="$REPO/results/full_mht_identity_history_scan_pruning_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
"$PY" -m bayescatrack benchmark suite \
  benchmarks/full_mht_identity_history_scan_pruning_manifest.json \
  --output-dir "$OUT" \
  --summary-format table

"$PY" -m bayescatrack.experiments.full_mht_identity_history_scan_pruning_decision \
  "$OUT/full_mht_identity_history_scan_pruning/full_mht_identity_history_scan_pruning_comparison.csv" \
  --output "$OUT/full_mht_identity_history_scan_pruning_decision.md"
```

## Exposure Audit

A scan-pruning score gain is not enough. The selected candidate weight must also
be audited without manual-GT labels, using the same combined identity-history
settings and the selected `--scan-motion-history-weight`.

Example for the central `0.50` weight:

```bash
EXPOSURE="$REPO/results/full_mht_identity_history_scan_pruning_exposure_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EXPOSURE"

"$PY" -m bayescatrack.experiments.track2p_policy_full_mht_exposure_audit \
  --data "$REPO/results/policy_dp/data_lightweight" \
  --input-format suite2p \
  --threshold-method min \
  --transform-type affine \
  --iou-distance-threshold 12 \
  --cell-probability-threshold 0.5 \
  --seed-session 0 \
  --beam-width 8 \
  --scan-hypotheses 8 \
  --edge-top-k 4 \
  --identity-diverse-beam \
  --miss-cost 2.0 \
  --full-mht-max-gap 1 \
  --gap-reactivation-cost 1.0 \
  --min-output-observations 1 \
  --min-edge-score 0.25 \
  --association-score-mode calibrated-likelihood \
  --association-likelihood-weight 1.0 \
  --association-likelihood-clip 4.0 \
  --track2p-prior-weight 12.0 \
  --track2p-non-prior-penalty 2.0 \
  --track2p-prior-switch-penalty 8.0 \
  --track2p-no-prior-successor-penalty 0.0 \
  --track2p-prior-miss-penalty 4.0 \
  --track2p-prior-survival-weight 1.0 \
  --track2p-prior-survival-min-examples-per-class 2 \
  --track2p-prior-survival-score-clip 8.0 \
  --no-prior-continuation-likelihood-weight 1.0 \
  --no-prior-continuation-min-examples-per-class 2 \
  --no-prior-continuation-score-clip 8.0 \
  --growth-history-prediction-weight 0.50 \
  --growth-history-prediction-scale 1.0 \
  --growth-history-prediction-clip 8.0 \
  --growth-history-prediction-min-edges 1 \
  --scan-motion-history-weight 0.50 \
  --output "$EXPOSURE/full_mht_identity_history_scan_pruning_exposure_050.csv" \
  --format csv \
  --progress
```

Inspect the `ALL` row fields:

```text
history_scan_motion_history_risk
history_scan_motion_history_weighted_risk
max_scan_motion_history_risk_per_subject
max_scan_motion_history_weighted_risk_per_subject
max_selected_non_prior_edges_per_subject
history_selected_non_prior_edges
```

The scan-pruning add-on remains exploratory if the exposure is broad, even when a
benchmark point improves.

## Interpretation

| result | meaning |
| --- | --- |
| `scan_pruning_stable_complete_history_gain` | candidate for integration into the paper-facing identity-history row, after the main identity-history promotion gates and exposure audit pass |
| `scan_pruning_single_weight_gain` | exploratory; the effect is too knife-edge |
| `scan_pruning_ties_identity_history` | useful method validation, not a better row |
| `scan_pruning_pairwise_only_gain` | not a complete-history result |
| `scan_pruning_regression_vs_identity_history` | do not promote; the add-on hurts the central row |
| `scan_pruning_beam_regression_vs_greedy` | do not promote; the full beam is worse than its matching greedy row |

This probe is deliberately separate from the current central candidate. It lets the method become more history-aware without silently moving the target after seeing a benchmark table.
