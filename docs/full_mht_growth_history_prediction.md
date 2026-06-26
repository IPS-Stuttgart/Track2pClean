# FullMHT Growth-History Prediction, 2026-06-26

This probe adds an opt-in dynamics term to the full scan-assignment MHT runner.
It is meant to move FullMHT further away from post-hoc cleanup and toward a
complete-track-aware Bayesian identity tracker.

## Method Layer

Base FullMHT scores a candidate edge from the current scan-pair diagnostics:
registered IoU, shifted IoU, area ratio, target cell probability, centroid
distance, growth residual, growth Mahalanobis residual, and local deformation.
Those terms are useful, but they are still local.

The growth-history prediction layer adds a row-history term while the scan cost
matrix is being built:

```text
edge_score_with_dynamics = local_edge_score
                           - growth_history_prediction_weight
                             * growth_history_prediction_penalty
```

For each candidate continuation, the scorer recovers the partial identity row
currently being expanded, parses that row's previous selected-edge summaries, and
compares the candidate's label-free diagnostics to the row's own history. A
candidate is penalized if it has abrupt deterioration in:

- registered IoU;
- shifted IoU;
- growth residual;
- growth Mahalanobis residual;
- local-neighborhood deformation.

This is not a terminal rerank and not a residual edit selector. The penalty is
applied before PyRecEst's Murty scan hypotheses are generated, so it can change
which full identity histories enter and survive the MHT beam.

## Frozen Probe

The immediate weight neighborhood is frozen in:

```text
benchmarks/full_mht_growth_history_prediction_probe_manifest.json
```

Rows:

| row | purpose |
| --- | --- |
| `Track2p` | original proposal baseline |
| `FullMHTPrior2` | proposal-prior FullMHT control |
| `FullMHTGrowthHistoryPrediction025` | growth-history prediction weight `0.25` |
| `FullMHTGrowthHistoryPrediction050` | growth-history prediction weight `0.50` |
| `FullMHTGrowthHistoryPrediction100` | growth-history prediction weight `1.00` |

All probe rows use:

```text
growth_history_prediction_scale = 1.0
growth_history_prediction_clip = 8.0
growth_history_prediction_min_edges = 1
```

## Validation Command

```bash
REPO=/home/florianpfaff/codex-runs/BayesCaTrack
PY="$REPO/.venv312/bin/python"
cd "$REPO"
git fetch origin
git checkout codex/full-mht-prototype
git reset --hard origin/codex/full-mht-prototype
export PYTHONPATH="$REPO/src"

"$PY" -m pytest -q \
  tests/test_full_mht_growth_history_prediction_integration.py \
  tests/test_full_mht_growth_history_prediction_decision.py \
  tests/test_full_mht_growth_history_prediction_promotion_gate.py \
  tests/test_full_mht_exposure_audit.py \
  tests/test_full_mht_no_gt_leakage.py \
  tests/test_benchmark_manifest_full_mht_integration.py

OUT="$REPO/results/full_mht_growth_history_prediction_probe_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
"$PY" -m bayescatrack benchmark suite \
  benchmarks/full_mht_growth_history_prediction_probe_manifest.json \
  --output-dir "$OUT" \
  --summary-format table

"$PY" -m bayescatrack.experiments.full_mht_growth_history_prediction_decision \
  "$OUT/full_mht_growth_history_prediction/full_mht_growth_history_prediction_comparison.csv" \
  --output "$OUT/full_mht_growth_history_prediction_decision.md"
```

## Exposure Audit

Benchmark metrics alone are not enough for this hook. After the manifest run,
record a label-free exposure table for the selected weight, starting with the
middle frozen setting:

```bash
EXPOSURE="$REPO/results/full_mht_growth_history_prediction_exposure_$(date +%Y%m%d_%H%M%S)"
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
  --track2p-prior-weight 12.0 \
  --track2p-non-prior-penalty 2.0 \
  --track2p-prior-switch-penalty 8.0 \
  --track2p-no-prior-successor-penalty 8.0 \
  --track2p-prior-miss-penalty 4.0 \
  --growth-history-prediction-weight 0.50 \
  --growth-history-prediction-scale 1.0 \
  --growth-history-prediction-clip 8.0 \
  --growth-history-prediction-min-edges 1 \
  --output "$EXPOSURE/full_mht_growth_history_prediction_exposure.csv" \
  --format csv \
  --progress
```

Inspect the `ALL` row fields:

```text
history_growth_prediction_evaluated_edges
history_growth_prediction_penalized_edges
history_growth_prediction_weighted_penalty
max_growth_prediction_penalized_edges_per_subject
max_growth_prediction_weighted_penalty_per_subject
```

A credible result should show bounded exposure: the penalty should be rare and
localized rather than suppressing many continuations across many subjects.

## Promotion Gate

The promotion gate combines the frozen benchmark comparison with the label-free
exposure table. It refuses promotion when the benchmark gain is single-weight, the
exposure audit was run without growth-history prediction enabled, or the penalty
fires broadly.

```bash
"$PY" -m bayescatrack.experiments.full_mht_growth_history_prediction_promotion_gate \
  "$OUT/full_mht_growth_history_prediction/full_mht_growth_history_prediction_comparison.csv" \
  "$EXPOSURE/full_mht_growth_history_prediction_exposure.csv" \
  --output "$EXPOSURE/full_mht_growth_history_prediction_promotion_gate.md"
```

Promotion requires:

```text
benchmark_result = history_dynamics_stable_gain
exposure_result  = bounded_exposure
```

If either condition fails, record the growth-history prediction layer as an
exploratory FullMHT component rather than a paper-facing benchmark row.

## Decision Rule

Treat this as a method probe, not a promoted row, until the manifest shows:

- pairwise F1 does not regress against `FullMHTPrior2`;
- complete-track F1 improves or at least does not regress;
- the effect is not limited to a single knife-edge weight;
- diagnostics show that the penalty is rare and targeted rather than suppressing
  broad continuation.

The frozen helper `full_mht_growth_history_prediction_decision.py` classifies the
probe as stable gain, single-weight gain, tie, pairwise regression, or complete
regression using the same no-tuning rule as the other FullMHT dynamics probes.
The promotion gate adds the required exposure check before any paper-facing row
can be promoted.

If it improves complete-track identity at more than one nearby weight, it becomes
a stronger FullMHT method component than a fixed prior-veto pocket. If it ties,
it remains useful as an architecture layer and constructed-conflict explanation,
but not as benchmark evidence.
