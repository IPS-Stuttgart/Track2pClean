# FullMHT Prior-Survival Validation Bundle, 2026-06-26

`FullMHTPriorSurvival` is the first FullMHT row that replaces the fixed
hand-gated prior-veto pocket with a calibrated, label-free prior-edge survival
likelihood. It should not be promoted from candidate row to paper method until it
passes the checks below.

The direct runner delegates to the base FullMHT implementation, installs the
calibrated survival scorer, and exposes `--track2p-prior-survival-*` knobs as
normal command-line flags:

```bash
"$PY" -m bayescatrack.experiments.track2p_policy_full_mht_prior_survival_benchmark \
  --help
```

The final promotion decision is deliberately mechanical. It combines three frozen
artifacts:

1. canonical manifest comparison with matching greedy ablations;
2. small prior-survival sensitivity table;
3. label-free exposure audit with prior-survival scoring enabled.

## Frozen Reproduction

Run the canonical manifest first:

```bash
REPO=/home/florianpfaff/codex-runs/BayesCaTrack
PY="$REPO/.venv312/bin/python"
cd "$REPO"
git fetch origin
git checkout codex/full-mht-prototype
git reset --hard origin/codex/full-mht-prototype
export PYTHONPATH="$REPO/src"

"$PY" -m pytest -q \
  tests/test_benchmark_manifest_full_mht_integration.py \
  tests/test_full_mht_manifest_decision.py \
  tests/test_full_mht_prior_survival_model.py \
  tests/test_full_mht_prior_survival_integration.py \
  tests/test_full_mht_prior_survival_runner.py \
  tests/test_full_mht_prior_survival_promotion_gate.py \
  tests/test_full_mht_exposure_audit.py \
  tests/test_full_mht_no_gt_leakage.py \
  tests/test_track2p_policy_full_mht_conflict_demo.py \
  tests/test_track2p_policy_full_mht_growth_prior.py::test_full_mht_prior_veto_scoring_does_not_read_gt_audit_columns

OUT="$REPO/results/full_mht_prior_survival_manifest_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
"$PY" -m bayescatrack benchmark suite \
  benchmarks/full_mht_prior_veto_manifest.json \
  --output-dir "$OUT" \
  --summary-format table

"$PY" -m bayescatrack.experiments.full_mht_manifest_decision \
  "$OUT/full_mht_prior_veto/full_mht_prior_veto_comparison.csv" \
  --output "$OUT/full_mht_manifest_decision.md"
```

Required rows:

| row | purpose |
| --- | --- |
| `Track2p` | original proposal baseline |
| `FullMHTPrior2` | FullMHT with strong Track2p proposal prior |
| `FullMHTGreedyPrior2` | greedy beam-width-1 ablation for the proposal-prior control |
| `FullMHTPriorVetoScaled` | fixed hand-gated prior-survival hazard |
| `FullMHTGreedyPriorVetoScaled` | greedy beam-width-1 ablation for the fixed hazard |
| `FullMHTPriorSurvival` | calibrated label-free prior-survival likelihood |
| `FullMHTGreedyPriorSurvival` | greedy beam-width-1 ablation for the calibrated survival row |

Promotion requires both manifest-decision conditions:

```text
history_search_result = prior_survival_complete_history_advantage
prior_survival_result = survival_improves_fixed_veto or survival_ties_fixed_veto
```

`prior_survival_complete_history_advantage` means the full calibrated-survival
beam improves complete-track F1 over `FullMHTGreedyPriorSurvival` without
pairwise-F1 loss. `fixed_veto_complete_history_advantage` is useful interim
evidence for the fixed-hazard row, but it does not by itself promote the
calibrated prior-survival candidate. A pairwise-only beam gain is not evidence for
the paper's complete-identity claim and must be recorded as exploratory.

## Direct Reproduction With Diagnostics

Use the explicit runner when diagnostics or summaries are needed for the survival
row itself:

```bash
DIRECT="$REPO/results/full_mht_prior_survival_direct_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DIRECT"

"$PY" -m bayescatrack.experiments.track2p_policy_full_mht_prior_survival_benchmark \
  --data "$REPO/results/policy_dp/data_lightweight" \
  --reference "$REPO/results/policy_dp/data_lightweight" \
  --reference-kind manual-gt \
  --input-format suite2p \
  --threshold-method min \
  --transform-type affine \
  --iou-distance-threshold 12 \
  --cell-probability-threshold 0.5 \
  --seed-source reference \
  --beam-width 8 \
  --scan-hypotheses 8 \
  --edge-top-k 4 \
  --identity-diverse-beam \
  --miss-cost 2.0 \
  --max-gap 1 \
  --gap-reactivation-cost 1.0 \
  --min-output-observations 1 \
  --min-edge-score 0.25 \
  --track2p-prior-weight 12.0 \
  --track2p-non-prior-penalty 2.0 \
  --track2p-prior-switch-penalty 8.0 \
  --track2p-no-prior-successor-penalty 8.0 \
  --track2p-prior-miss-penalty 4.0 \
  --track2p-prior-survival-weight 1.0 \
  --track2p-prior-survival-min-examples-per-class 2 \
  --track2p-prior-survival-score-clip 8.0 \
  --output "$DIRECT/full_mht_prior_survival.csv" \
  --format csv \
  --diagnostics-output "$DIRECT/diagnostics.csv" \
  --diagnostics-format csv \
  --summary-output "$DIRECT/summary.csv" \
  --progress
```

## Sensitivity Table

Run the frozen neighborhood manifest next:

```bash
SENS="$REPO/results/full_mht_prior_survival_sensitivity_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SENS"
"$PY" -m bayescatrack benchmark suite \
  benchmarks/full_mht_prior_survival_sensitivity_manifest.json \
  --output-dir "$SENS" \
  --summary-format table
```

The manifest varies only immediate method-neighborhood settings:

| factor | values |
| --- | --- |
| survival weight | `0.5`, `1.0`, `1.5` |
| survival score clip | `4.0`, `8.0` |
| minimum pseudo examples per class | `2`, `3` |
| anchor strictness | default vs stricter anchor overlap/confidence |

The promotion gate requires `stable_plateau`: the central row must pass, at least
four of the six sensitivity rows must stay at or above `FullMHTPrior2` on
pairwise and complete-track F1, and at least two of the three weight-neighborhood
rows must pass. Pairwise collapse in any row keeps the layer exploratory.

## Non-GT Exposure Audit

The exposure audit is deliberately not a benchmark and does not load reference
labels. It uses Track2p output only as the seed/proposal source and records how
broadly the FullMHT candidate layer fires across all Track2p-style subjects.

```bash
AUDIT="$REPO/results/full_mht_prior_survival_exposure_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$AUDIT"

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
  --track2p-prior-survival-weight 1.0 \
  --track2p-prior-survival-min-examples-per-class 2 \
  --track2p-prior-survival-score-clip 8.0 \
  --output "$AUDIT/prior_survival_exposure.csv" \
  --format csv \
  --progress
```

The audit must include prior-survival exposure columns such as
`history_prior_survival_scored_edges`, `history_prior_survival_negative_edges`,
and `max_prior_survival_negative_edges_per_subject`. If these are missing or the
scored-edge count is zero, the promotion gate reports the exposure artifact as
incomplete.

## Combined Promotion Gate

After the canonical manifest, sensitivity manifest, and exposure audit exist, run:

```bash
"$PY" -m bayescatrack.experiments.full_mht_prior_survival_promotion_gate \
  "$OUT/full_mht_prior_veto/full_mht_prior_veto_comparison.csv" \
  "$SENS/full_mht_prior_survival_sensitivity/full_mht_prior_survival_sensitivity.csv" \
  "$AUDIT/prior_survival_exposure.csv" \
  --output "$AUDIT/prior_survival_promotion_gate.md"
```

Promotion requires:

- `manifest_result = prior_survival_complete_history_advantage`;
- `prior_survival_result = survival_improves_fixed_veto` or `survival_ties_fixed_veto`;
- `sensitivity_result = stable_plateau`;
- `exposure_result = bounded_exposure`;
- prior-survival scored edges are nonzero;
- selected non-prior edges, prior switches, no-prior continuations, and negative
  prior-survival penalties remain bounded.

If any gate fails, keep the row exploratory and record the failure reason. This is
the safeguard that prevents a ledger-discovered prior-survival likelihood from
being presented as a validated method row.

## Recording

After the server runs, update this document and
`docs/full_mht_manifest_integration_notes.md` with:

- output directories;
- focused pytest result;
- canonical comparison table;
- `full_mht_manifest_decision.md`;
- direct diagnostic run summary;
- sensitivity table;
- exposure counts table from `prior_survival_exposure.csv`;
- `prior_survival_promotion_gate.md`;
- promote / keep exploratory / reject decision.
