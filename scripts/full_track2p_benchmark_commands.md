# Full Track2p Validate-Diagnose-Benchmark Command Recipe

Use this recipe after assembling the raw pre-Track2p Suite2p subject tree and the independent manual-GT reference root. The recipe is intentionally split into three tiers:

1. **validation gates** that should pass before reporting numbers,
2. **diagnostic/oracle exports** that explain where BayesCaTrack loses edges, and
3. **paper-facing benchmark rows** that can be aggregated into the comparison table and figure.

The commands below assume BayesCaTrack is installed from the current code repo, for example with:

```bash
python -m pip install -e /path/to/BayesCaTrack[track2p]
```

## 0. Set paths and benchmark knobs

```bash
DATA_ROOT=/path/to/full/pre-track2p-suite2p-root
REFERENCE_ROOT=/path/to/manual-gt-root
TRACK2P_REFERENCE_ROOT=/path/to/track2p-output-root-or-data-root
RESULTS_DIR=results/full_track2p

MAX_GAP=2
CORE_TRANSFORM_TYPE=affine
FOV_TRANSFORM_TYPE=fov-affine

# Current best tuned solver row reported in the working notes.
COST_SCALE=1.25
COST_THRESHOLD=2
START_COST=1
END_COST=1
GAP_PENALTY=0.6

mkdir -p "$RESULTS_DIR"/{validation,oracles,registration,edge_ranking,teacher,benchmarks}

COMMON_ARGS=(
  --data "$DATA_ROOT"
  --reference "$REFERENCE_ROOT"
  --reference-kind manual-gt
  --input-format suite2p
  --include-non-cells
  --no-include-behavior
)

SOLVER_ARGS=(
  --max-gap "$MAX_GAP"
  --start-cost "$START_COST"
  --end-cost "$END_COST"
  --gap-penalty "$GAP_PENALTY"
  --cost-threshold "$COST_THRESHOLD"
)
```

`CORE_TRANSFORM_TYPE` is used for commands whose local CLI currently exposes the core registration choices (`affine`, `rigid`, `fov-translation`, `none`). The tuned FOV-affine row is run through the cost-sweep path, which exposes `fov-affine` and cost scaling.

## 1. Validate raw Suite2p/manual-GT compatibility

These commands write artifacts even when there is an incompatibility, so the diagnostics are preserved. Do not continue to paper-facing metrics until the Markdown summaries report compatible manual-GT ROI coverage.

```bash
bayescatrack benchmark validate-track2p-inputs \
  "${COMMON_ARGS[@]}" \
  --no-fail-on-incompatible \
  --format markdown \
  --output "$RESULTS_DIR/validation/full_suite2p_roi_validation.md"

bayescatrack benchmark validate-track2p-inputs \
  "${COMMON_ARGS[@]}" \
  --no-fail-on-incompatible \
  --format csv \
  --output "$RESULTS_DIR/validation/full_suite2p_roi_validation.csv"

bayescatrack benchmark audit-manual-gt-rois \
  "${COMMON_ARGS[@]}" \
  --format markdown \
  --output "$RESULTS_DIR/validation/manual_gt_roi_index_audit.md"

bayescatrack benchmark audit-manual-gt-rois \
  "${COMMON_ARGS[@]}" \
  --format csv \
  --output "$RESULTS_DIR/validation/manual_gt_roi_index_audit.csv"
```

## 2. Run row-stitching and registration oracles

The `oracle-gt-links` row should be close to perfect on compatible manual-GT subjects. If it is not, the bug is in row stitching, ROI-index mapping, or scoring rather than in association costs.

```bash
bayescatrack benchmark track2p \
  "${COMMON_ARGS[@]}" \
  --method oracle-gt-links \
  --format csv \
  --output "$RESULTS_DIR/oracles/oracle_gt_links.csv"
```

Registration diagnostics should be kept as supporting artifacts even if they are not part of the headline comparison.

```bash
bayescatrack benchmark registration-qa \
  "${COMMON_ARGS[@]}" \
  --cost registered-iou \
  --transform-type "$CORE_TRANSFORM_TYPE" \
  --max-gap "$MAX_GAP" \
  --level summary \
  --format csv \
  --output "$RESULTS_DIR/registration/registration_${CORE_TRANSFORM_TYPE}_summary.csv"

bayescatrack benchmark registration-qa \
  "${COMMON_ARGS[@]}" \
  --cost registered-iou \
  --transform-type "$CORE_TRANSFORM_TYPE" \
  --max-gap "$MAX_GAP" \
  --level links \
  --format csv \
  --output "$RESULTS_DIR/registration/registration_${CORE_TRANSFORM_TYPE}_links.csv"

bayescatrack benchmark oracle-affine-qa \
  "${COMMON_ARGS[@]}" \
  --cost registered-iou \
  --transform-type "$CORE_TRANSFORM_TYPE" \
  --max-gap "$MAX_GAP" \
  --level summary \
  --format csv \
  --output "$RESULTS_DIR/registration/oracle_affine_summary.csv"

bayescatrack benchmark oracle-affine-qa \
  "${COMMON_ARGS[@]}" \
  --cost registered-iou \
  --transform-type "$CORE_TRANSFORM_TYPE" \
  --max-gap "$MAX_GAP" \
  --level links \
  --format csv \
  --output "$RESULTS_DIR/registration/oracle_affine_links.csv"

bayescatrack benchmark growth-registration-qa \
  "${COMMON_ARGS[@]}" \
  --cost registered-iou \
  --transform-type gt-affine-oracle \
  --max-gap "$MAX_GAP" \
  --level spatial-summary \
  --format csv \
  --output "$RESULTS_DIR/registration/growth_spatial_summary.csv"
```

## 3. Export edge-ranking diagnostics before global assignment

These are the most important diagnostics for closing the Track2p gap. They tell whether manual-GT edges are missing, present but ranked too low, or already high-ranked but later rejected by the solver.

```bash
bayescatrack benchmark edge-ranking \
  "${COMMON_ARGS[@]}" \
  --cost registered-iou \
  --transform-type "$CORE_TRANSFORM_TYPE" \
  --max-gap "$MAX_GAP" \
  --output "$RESULTS_DIR/edge_ranking/registered_iou_edge_ranking.csv" \
  --summary-output "$RESULTS_DIR/edge_ranking/registered_iou_edge_ranking_summary.csv"

bayescatrack benchmark edge-ranking \
  "${COMMON_ARGS[@]}" \
  --cost roi-aware \
  --transform-type "$CORE_TRANSFORM_TYPE" \
  --max-gap "$MAX_GAP" \
  --weighted-masks \
  --weighted-centroids \
  --output "$RESULTS_DIR/edge_ranking/roi_aware_edge_ranking.csv" \
  --summary-output "$RESULTS_DIR/edge_ranking/roi_aware_edge_ranking_summary.csv"
```

## 4. Export Track2p-as-teacher/debug-oracle artifacts

Use these only for analysis and error localization. The paper-facing score reference remains the independent manual GT.

```bash
bayescatrack benchmark track2p-teacher-audit \
  --data "$DATA_ROOT" \
  --ground-truth-reference "$REFERENCE_ROOT" \
  --track2p-reference "$TRACK2P_REFERENCE_ROOT" \
  --input-format suite2p \
  --include-non-cells \
  --no-include-behavior \
  --cost registered-iou \
  --transform-type "$CORE_TRANSFORM_TYPE" \
  "${SOLVER_ARGS[@]}" \
  --format csv \
  --output "$RESULTS_DIR/teacher/teacher_audit_summary.csv" \
  --edges-output "$RESULTS_DIR/teacher/teacher_edges.csv" \
  --focus-output "$RESULTS_DIR/teacher/teacher_focus_bayes_missed_track2p.csv" \
  --teacher-output "$RESULTS_DIR/teacher/teacher_training_rows.csv"

bayescatrack benchmark track2p-teacher-debug \
  "${COMMON_ARGS[@]}" \
  --track2p-reference "$TRACK2P_REFERENCE_ROOT" \
  --cost registered-iou \
  --transform-type "$CORE_TRANSFORM_TYPE" \
  "${SOLVER_ARGS[@]}" \
  --cost-scale "$COST_SCALE" \
  --edge-scope solver \
  --details-output "$RESULTS_DIR/teacher/teacher_debug_details.csv" \
  --summary-output "$RESULTS_DIR/teacher/teacher_debug_summary.csv" \
  --metrics-output "$RESULTS_DIR/teacher/teacher_debug_metrics.csv"
```

## 5. Run benchmark rows

### 5.1 Track2p baseline and simple BayesCaTrack baselines

```bash
bayescatrack benchmark track2p \
  "${COMMON_ARGS[@]}" \
  --method track2p-baseline \
  --format csv \
  --output "$RESULTS_DIR/benchmarks/track2p_baseline.csv"

bayescatrack benchmark track2p \
  "${COMMON_ARGS[@]}" \
  --method global-assignment \
  --cost registered-iou \
  --transform-type "$CORE_TRANSFORM_TYPE" \
  --max-gap 1 \
  --format csv \
  --output "$RESULTS_DIR/benchmarks/global_registered_iou_gap1.csv"

bayescatrack benchmark track2p \
  "${COMMON_ARGS[@]}" \
  --method global-assignment \
  --cost registered-iou \
  --transform-type "$CORE_TRANSFORM_TYPE" \
  --max-gap "$MAX_GAP" \
  --format csv \
  --output "$RESULTS_DIR/benchmarks/global_registered_iou_gap${MAX_GAP}.csv"

bayescatrack benchmark track2p \
  "${COMMON_ARGS[@]}" \
  --method global-assignment \
  --cost roi-aware \
  --transform-type "$CORE_TRANSFORM_TYPE" \
  --max-gap "$MAX_GAP" \
  --weighted-masks \
  --weighted-centroids \
  --format csv \
  --output "$RESULTS_DIR/benchmarks/global_roi_aware_gap${MAX_GAP}.csv"
```

### 5.2 Current tuned FOV-affine registered-IoU row

Use the sweep runner for this row because it applies the explicit cost scale used in the current best configuration.

```bash
bayescatrack benchmark track2p-sweep \
  "${COMMON_ARGS[@]}" \
  --cost registered-iou \
  --transform-type "$FOV_TRANSFORM_TYPE" \
  --max-gap "$MAX_GAP" \
  --start-cost "$START_COST" \
  --end-cost "$END_COST" \
  --gap-penalty "$GAP_PENALTY" \
  --cost-scales "$COST_SCALE" \
  --cost-thresholds "$COST_THRESHOLD" \
  --write-incrementally \
  --format csv \
  --output "$RESULTS_DIR/benchmarks/global_registered_iou_fov_affine_tuned_gap${MAX_GAP}.csv"
```

### 5.3 Solver-prior LOSO, calibrated LOSO, and monotone-ranking LOSO

```bash
bayescatrack benchmark track2p-solver-prior-loso \
  "${COMMON_ARGS[@]}" \
  --cost registered-iou \
  --transform-type "$CORE_TRANSFORM_TYPE" \
  --max-gap "$MAX_GAP" \
  --objective pairwise_f1 \
  --start-costs "0.5,1,2" \
  --end-costs "0.5,1,2" \
  --gap-penalties "0,0.6,1.2" \
  --cost-thresholds "1.5,2,4,none" \
  --format csv \
  --output "$RESULTS_DIR/benchmarks/global_registered_iou_tuned_priors_gap${MAX_GAP}.csv"

bayescatrack benchmark track2p-solver-prior-loso \
  "${COMMON_ARGS[@]}" \
  --cost roi-aware \
  --transform-type "$CORE_TRANSFORM_TYPE" \
  --max-gap "$MAX_GAP" \
  --weighted-masks \
  --weighted-centroids \
  --objective pairwise_f1 \
  --start-costs "0.5,1,2" \
  --end-costs "0.5,1,2" \
  --gap-penalties "0,0.6,1.2" \
  --cost-thresholds "1.5,2,4,none" \
  --format csv \
  --output "$RESULTS_DIR/benchmarks/global_roi_aware_tuned_priors_gap${MAX_GAP}.csv"

bayescatrack benchmark track2p \
  "${COMMON_ARGS[@]}" \
  --method global-assignment \
  --cost calibrated \
  --split leave-one-subject-out \
  --transform-type "$CORE_TRANSFORM_TYPE" \
  --max-gap "$MAX_GAP" \
  --format csv \
  --output "$RESULTS_DIR/benchmarks/global_calibrated_loso_gap${MAX_GAP}.csv"

bayescatrack benchmark track2p-monotone-loso \
  "${COMMON_ARGS[@]}" \
  --transform-type "$CORE_TRANSFORM_TYPE" \
  "${SOLVER_ARGS[@]}" \
  --monotone-ranker-kwargs-json '{"max_negatives_per_positive":32,"margin":1.0}' \
  --format csv \
  --output "$RESULTS_DIR/benchmarks/global_monotone_loso_gap${MAX_GAP}.csv"
```

## 6. Aggregate paper-facing and diagnostic comparisons

The paper-facing comparison excludes oracle rows. The diagnostic comparison includes the GT-link oracle so row-stitching/scoring failures are visible.

```bash
bayescatrack benchmark compare \
  --input Track2p="$RESULTS_DIR/benchmarks/track2p_baseline.csv" \
  --input Global-IoU-gap1="$RESULTS_DIR/benchmarks/global_registered_iou_gap1.csv" \
  --input Global-IoU-gap${MAX_GAP}="$RESULTS_DIR/benchmarks/global_registered_iou_gap${MAX_GAP}.csv" \
  --input ROI-aware-gap${MAX_GAP}="$RESULTS_DIR/benchmarks/global_roi_aware_gap${MAX_GAP}.csv" \
  --input FOV-affine-tuned="$RESULTS_DIR/benchmarks/global_registered_iou_fov_affine_tuned_gap${MAX_GAP}.csv" \
  --input Global-IoU-LOSO-priors="$RESULTS_DIR/benchmarks/global_registered_iou_tuned_priors_gap${MAX_GAP}.csv" \
  --input ROI-aware-LOSO-priors="$RESULTS_DIR/benchmarks/global_roi_aware_tuned_priors_gap${MAX_GAP}.csv" \
  --input Calibrated-LOSO="$RESULTS_DIR/benchmarks/global_calibrated_loso_gap${MAX_GAP}.csv" \
  --input Monotone-LOSO="$RESULTS_DIR/benchmarks/global_monotone_loso_gap${MAX_GAP}.csv" \
  --highlight-best \
  --include-best-summary \
  --output "$RESULTS_DIR/benchmarks/comparison.md"

bayescatrack benchmark compare \
  --input Track2p="$RESULTS_DIR/benchmarks/track2p_baseline.csv" \
  --input Global-IoU-gap1="$RESULTS_DIR/benchmarks/global_registered_iou_gap1.csv" \
  --input Global-IoU-gap${MAX_GAP}="$RESULTS_DIR/benchmarks/global_registered_iou_gap${MAX_GAP}.csv" \
  --input ROI-aware-gap${MAX_GAP}="$RESULTS_DIR/benchmarks/global_roi_aware_gap${MAX_GAP}.csv" \
  --input FOV-affine-tuned="$RESULTS_DIR/benchmarks/global_registered_iou_fov_affine_tuned_gap${MAX_GAP}.csv" \
  --input Global-IoU-LOSO-priors="$RESULTS_DIR/benchmarks/global_registered_iou_tuned_priors_gap${MAX_GAP}.csv" \
  --input ROI-aware-LOSO-priors="$RESULTS_DIR/benchmarks/global_roi_aware_tuned_priors_gap${MAX_GAP}.csv" \
  --input Calibrated-LOSO="$RESULTS_DIR/benchmarks/global_calibrated_loso_gap${MAX_GAP}.csv" \
  --input Monotone-LOSO="$RESULTS_DIR/benchmarks/global_monotone_loso_gap${MAX_GAP}.csv" \
  --format csv \
  --output "$RESULTS_DIR/benchmarks/comparison.csv"

bayescatrack benchmark compare \
  --input Oracle-GT-links="$RESULTS_DIR/oracles/oracle_gt_links.csv" \
  --input Track2p="$RESULTS_DIR/benchmarks/track2p_baseline.csv" \
  --input FOV-affine-tuned="$RESULTS_DIR/benchmarks/global_registered_iou_fov_affine_tuned_gap${MAX_GAP}.csv" \
  --input Monotone-LOSO="$RESULTS_DIR/benchmarks/global_monotone_loso_gap${MAX_GAP}.csv" \
  --highlight-best \
  --include-best-summary \
  --output "$RESULTS_DIR/benchmarks/comparison_with_oracles.md"

bayescatrack benchmark compare \
  --input Oracle-GT-links="$RESULTS_DIR/oracles/oracle_gt_links.csv" \
  --input Track2p="$RESULTS_DIR/benchmarks/track2p_baseline.csv" \
  --input FOV-affine-tuned="$RESULTS_DIR/benchmarks/global_registered_iou_fov_affine_tuned_gap${MAX_GAP}.csv" \
  --input Monotone-LOSO="$RESULTS_DIR/benchmarks/global_monotone_loso_gap${MAX_GAP}.csv" \
  --format csv \
  --output "$RESULTS_DIR/benchmarks/comparison_with_oracles.csv"
```

## 7. Generate the paper figure

Run this from the paper repository after either keeping `RESULTS_DIR` inside the repo or copying the CSV with `scripts/copy_bayescatrack_artifacts.sh`.

```bash
python scripts/plot_benchmark_comparison.py \
  --input "$RESULTS_DIR/benchmarks/comparison.csv" \
  --output figures/benchmark_comparison.svg
```

## 8. Optional GitHub Actions shortcut

The BayesCaTrack code repo also has a manual `Track2p raw Suite2p benchmark` workflow. Prefer the workflow for long runs, large cost sweeps, or non-core registration backends. Keep this paper-repo recipe as the reproducible command ledger for the exact artifacts copied into the paper repository.
