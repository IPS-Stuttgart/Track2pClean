# FullMHT Prior-Survival Validation Bundle, 2026-06-26

`FullMHTPriorSurvival` is the first FullMHT row that replaces the fixed
hand-gated prior-veto pocket with a calibrated, label-free prior-edge survival
likelihood. It should not be promoted from candidate row to paper method until it
passes the checks below.

The row can be run either through the benchmark manifest or directly with:

```bash
"$PY" -m bayescatrack.experiments.track2p_policy_full_mht_prior_survival_benchmark \
  --help
```

The direct runner delegates to the base FullMHT implementation, installs the
calibrated survival scorer, and exposes `--track2p-prior-survival-*` knobs as
normal command-line flags.

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
| `FullMHTGreedyPrior2` | greedy beam-width-1 ablation over the same scan candidates |
| `FullMHTPriorVetoScaled` | fixed hand-gated prior-survival hazard |
| `FullMHTPriorSurvival` | calibrated label-free prior-survival likelihood |

Promotion requires both of these manifest-decision conditions:

```text
history_search_result = beam_complete_history_advantage
prior_survival_result = survival_improves_fixed_veto or survival_ties_fixed_veto
```

`beam_complete_history_advantage` means the full beam improves complete-track F1
over `FullMHTGreedyPrior2` without pairwise-F1 loss. A pairwise-only beam gain is
not evidence for the paper's complete-identity claim and must be recorded as
exploratory.

The decision artifact reports:

- whether the beam row gives a complete-track advantage, ties the greedy row,
  regresses against it, or improves only pairwise F1;
- whether `FullMHTPriorSurvival` improves, ties, or falls below
  `FullMHTPriorVetoScaled`;
- the pairwise/complete-track micro deltas used for the decision.

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
OUT="$REPO/results/full_mht_prior_survival_sensitivity_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
"$PY" -m bayescatrack benchmark suite \
  benchmarks/full_mht_prior_survival_sensitivity_manifest.json \
  --output-dir "$OUT" \
  --summary-format table
```

The manifest varies only immediate method-neighborhood settings:

| factor | values |
| --- | --- |
| survival weight | `0.5`, `1.0`, `1.5` |
| survival score clip | `4.0`, `8.0` |
| minimum pseudo examples per class | `2`, `3` |
| anchor strictness | default vs stricter anchor overlap/confidence |

Decision rule:

- complete-track F1 should stay at least as high as `FullMHTPrior2` for nearby
  settings;
- pairwise F1 should not collapse in any immediate neighbor;
- if only one exact setting works, report the survival layer as exploratory;
- if a small plateau works, `FullMHTPriorSurvival` can replace the fixed
  prior-veto hazard as the stronger method row.

## Non-GT Exposure Audit

The exposure audit is deliberately not a manual-GT benchmark. It runs the same
FullMHT prior-survival configuration with Track2p output as the reference/seed
source so that all Track2p-style subjects can be inspected for broad firing.
The scoring numbers in this audit are not paper metrics; the summary counts are
the point.

```bash
AUDIT="$REPO/results/full_mht_prior_survival_exposure_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$AUDIT"

"$PY" -m bayescatrack.experiments.track2p_policy_full_mht_prior_survival_benchmark \
  --data "$REPO/results/policy_dp/data_lightweight" \
  --reference "$REPO/results/policy_dp/data_lightweight" \
  --reference-kind track2p-output \
  --allow-track2p-as-reference-for-smoke-test \
  --input-format suite2p \
  --threshold-method min \
  --transform-type affine \
  --iou-distance-threshold 12 \
  --cell-probability-threshold 0.5 \
  --seed-source track2p-output \
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
  --output "$AUDIT/scores_against_track2p_reference.csv" \
  --format csv \
  --diagnostics-output "$AUDIT/diagnostics.csv" \
  --diagnostics-format csv \
  --summary-output "$AUDIT/summary.csv" \
  --progress
```

Decision rule:

- `scan_selected_non_prior_edges` should remain rare;
- `scan_missed_prior_successors` should remain tiny, not broad across subjects;
- no subject should receive a large number of prior switches or no-prior
  continuations;
- if exposure is broad, the survival model is too permissive or too strong even
  if the manual-GT benchmark improves.

## Recording

After the server runs, update this document and
`docs/full_mht_manifest_integration_notes.md` with:

- output directories;
- focused pytest result;
- canonical comparison table;
- `full_mht_manifest_decision.md`;
- direct diagnostic run summary;
- sensitivity table;
- exposure counts table from `summary.csv`;
- promote / keep exploratory / reject decision.
