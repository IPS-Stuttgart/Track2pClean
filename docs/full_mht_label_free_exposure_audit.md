# FullMHT Label-Free Exposure Audit, 2026-06-26

`track2p_policy_full_mht_exposure_audit.py` runs the FullMHT scan-assignment beam
without loading manual-GT references. It starts from Track2p output seed tracks,
uses Track2p proposal edges as priors, and writes per-subject behavior counts.

This is an exposure audit, not a benchmark scorer. It answers whether an opt-in
FullMHT method layer fires rarely and locally, or whether it broadly changes many
subjects.

## Why This Exists

The official benchmark runner must load manual-GT to compute F1. That makes it
unsuitable for non-GT exposure checks. The exposure audit bypasses scoring and
reports label-free quantities only:

- output track count;
- selected prior and non-prior edge counts;
- missed tracks and missed prior successors;
- prior switches and no-prior continuations;
- gap reactivations;
- missing observations in selected histories;
- terminal identity and motion-history risks when those hooks are enabled;
- growth-history prediction evaluated and penalized edge counts when that hook
  is enabled;
- total and per-subject growth-history prediction penalty mass.

## Run

```bash
REPO=/home/florianpfaff/codex-runs/BayesCaTrack
PY="$REPO/.venv312/bin/python"
cd "$REPO"
git fetch origin
git checkout codex/full-mht-prototype
git reset --hard origin/codex/full-mht-prototype
export PYTHONPATH="$REPO/src"

"$PY" -m pytest -q \
  tests/test_full_mht_no_gt_leakage.py \
  tests/test_full_mht_exposure_audit.py \
  tests/test_full_mht_history_dynamics_promotion_gate.py

OUT="$REPO/results/full_mht_label_free_exposure_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

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
  --track2p-prior-weight 12.0 \
  --track2p-non-prior-penalty 2.0 \
  --track2p-prior-switch-penalty 8.0 \
  --track2p-no-prior-successor-penalty 8.0 \
  --track2p-prior-miss-penalty 4.0 \
  --terminal-motion-history-weight 0.50 \
  --output "$OUT/full_mht_history_dynamics_exposure.csv" \
  --format csv \
  --progress
```

## Growth-History Exposure Variant

Run this variant after the growth-history prediction manifest. It uses the same
label-free exposure runner, but installs the scan-time growth-history scoring hook
and records how often it actually penalizes selected continuations.

```bash
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
  --track2p-prior-weight 12.0 \
  --track2p-non-prior-penalty 2.0 \
  --track2p-prior-switch-penalty 8.0 \
  --track2p-no-prior-successor-penalty 8.0 \
  --track2p-prior-miss-penalty 4.0 \
  --growth-history-prediction-weight 0.50 \
  --growth-history-prediction-scale 1.0 \
  --growth-history-prediction-clip 8.0 \
  --growth-history-prediction-min-edges 1 \
  --output "$OUT/full_mht_growth_history_prediction_exposure.csv" \
  --format csv \
  --progress
```

## Combined Gate

After running `benchmarks/full_mht_history_dynamics_probe_manifest.json`, combine
that benchmark sensitivity table with the exposure audit:

```bash
PROBE="$REPO/results/full_mht_history_dynamics_probe_YYYYMMDD_HHMMSS"

"$PY" -m bayescatrack.experiments.full_mht_history_dynamics_promotion_gate \
  "$PROBE/full_mht_history_dynamics/full_mht_history_dynamics_comparison.csv" \
  "$OUT/full_mht_history_dynamics_exposure.csv" \
  --output "$OUT/full_mht_history_dynamics_promotion_gate.md"
```

Promotion requires:

```text
benchmark_result = history_dynamics_stable_gain
exposure_result  = bounded_exposure
```

If the benchmark improves but exposure is broad, keep the row exploratory.

## Readout

The `ALL` row should be inspected first:

```text
max_selected_non_prior_edges_per_subject
history_selected_non_prior_edges
history_switched_prior_successors
history_no_prior_successor_continuations
history_gap_reactivated_tracks
max_missing_observations_per_subject
history_growth_prediction_evaluated_edges
history_growth_prediction_penalized_edges
history_growth_prediction_weighted_penalty
max_growth_prediction_penalized_edges_per_subject
max_growth_prediction_weighted_penalty_per_subject
```

Healthy exposure means changes remain rare and no subject receives a large number
of non-prior continuations, switches, or growth-history penalties. If the audit
shows broad firing, keep the method layer exploratory even if a manual-GT
benchmark row improves.

## Leakage Guard

`tests/test_full_mht_no_gt_leakage.py` covers the method hooks and this exposure
audit runner. It fails if selector/audit code references manual-GT loaders,
benchmark scorers, or audit-result columns such as `edge_status_against_gt`,
`pairwise_delta_if_removed`, or `complete_delta_if_removed`.
