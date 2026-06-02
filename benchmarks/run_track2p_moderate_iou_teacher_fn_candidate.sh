#!/usr/bin/env bash
set -euo pipefail

# Focused Track2p-teacher adjacent-FN rescue candidate.
#
# Motivation:
#   The Track2p-policy emulation ledger showed that the remaining Track2p-only
#   true-positive adjacent edges had moderate registered IoU values rather than
#   very high overlap, while previous broad teacher/gap rescue rows admitted too
#   many false positives or changed no official counts. This script therefore
#   starts from the frozen ComponentCleanup row and tests one narrow, label-free
#   rescue hypothesis:
#
#     Use Track2p only as a teacher-edge candidate generator for adjacent target
#     extensions, require moderate registered IoU, strong cell probabilities,
#     and existing component support, and cap accepted edits per subject.
#
# The row is a Track2p-teacher hybrid ablation, not an independent tracker row.
# It should be promoted only if it beats ComponentCleanup without degrading the
# complete-track objective.
#
# Required:
#   DATA=/path/to/track2p-style/data-or-resolved-lightweight-view
#
# Optional:
#   REF=/path/to/manual-gt-reference-root  # defaults to DATA
#   REPO=/path/to/BayesCaTrack            # defaults to current directory
#   OUT=/path/to/output-dir               # defaults under $REPO/results
#   PY=/path/to/python                    # defaults to python

REPO=${REPO:-$(pwd)}
DATA=${DATA:?Set DATA to the Track2p-style data root or resolved lightweight view}
REF=${REF:-$DATA}
PY=${PY:-python}
OUT=${OUT:-"$REPO/results/moderate_iou_teacher_fn_candidate_$(date +%Y%m%d_%H%M%S)"}

mkdir -p "$OUT"
exec > >(tee "$OUT/run.log") 2>&1

cd "$REPO"
export PYTHONPATH=${PYTHONPATH:-"$REPO/src"}

echo "started_at=$(date --iso-8601=seconds)"
echo "repo=$REPO"
echo "data=$DATA"
echo "reference=$REF"
echo "out=$OUT"
echo "python=$PY"

git rev-parse HEAD | tee "$OUT/git_sha.txt" || true
git status --short | tee "$OUT/git_status_before.txt" || true
"$PY" -m pip freeze > "$OUT/pip_freeze.txt" || true

COMMON=(
  --data "$DATA"
  --reference "$REF"
  --reference-kind manual-gt
  --input-format suite2p
  --plane plane0
  --no-include-behavior
)
POLICY=(
  --threshold-method min
  --transform-type affine
  --iou-distance-threshold 12
  --cell-probability-threshold 0.5
)
CLEANUP=(
  --split-risk-threshold 1.5
  --min-side-observations 2
)

run_teacher_rescue() {
  local label=$1
  shift
  "$PY" -m bayescatrack benchmark track2p-policy-teacher-adjacent-rescue \
    "${COMMON[@]}" \
    "${POLICY[@]}" \
    "${CLEANUP[@]}" \
    "$@" \
    --output "$OUT/${label}.csv" \
    --format csv \
    --diagnostics-output "$OUT/${label}_edges.csv" \
    --diagnostics-format csv
}

"$PY" -m bayescatrack benchmark track2p \
  "${COMMON[@]}" \
  --method track2p-baseline \
  --output "$OUT/track2p_baseline.csv" \
  --format csv

"$PY" -m bayescatrack benchmark track2p-policy \
  "${COMMON[@]}" \
  "${POLICY[@]}" \
  --output "$OUT/track2p_policy_d12.csv" \
  --format csv

"$PY" -m bayescatrack benchmark track2p-policy-component-audit \
  "${COMMON[@]}" \
  "${POLICY[@]}" \
  --apply-splits \
  "${CLEANUP[@]}" \
  --output "$OUT/track2p_policy_component_cleanup.csv" \
  --format csv \
  --component-output "$OUT/track2p_policy_component_cleanup_components.csv" \
  --component-format csv

# Existing residual-FN candidate row for comparison.
run_teacher_rescue teacher_adjacent_dynamic_confidence_residual_fn_cell_confident_max2 \
  --no-allow-completing-rescue \
  --allow-source-backfill \
  --no-allow-seed-source-backfill \
  --allow-fragment-merges \
  --teacher-edge-order dynamic-confidence \
  --teacher-feature-preset residual-fn-cell-confident \
  --min-component-observations 2 \
  --max-applied-edits 2

# New focused candidate: adjacent target-extension rescue, moderate-IoU window,
# cell confidence, and a small edit cap. The max-IoU gate avoids the high-overlap
# teacher-edge tail that did not explain the known Track2p-only TP residuals.
run_teacher_rescue teacher_adjacent_moderate_iou_cell_target_extension_max3 \
  --no-allow-completing-rescue \
  --allow-source-backfill \
  --no-allow-seed-source-backfill \
  --allow-fragment-merges \
  --teacher-action-filter target-extension \
  --teacher-edge-order dynamic-confidence \
  --teacher-min-registered-iou 0.10 \
  --teacher-max-registered-iou 0.55 \
  --teacher-max-centroid-distance 6.0 \
  --teacher-min-area-ratio 0.45 \
  --teacher-min-cell-probability 0.60 \
  --min-component-observations 2 \
  --max-applied-edits 3

"$PY" -m bayescatrack benchmark compare \
  --input Track2p="$OUT/track2p_baseline.csv" \
  --input Track2pPolicyD12="$OUT/track2p_policy_d12.csv" \
  --input ComponentCleanup="$OUT/track2p_policy_component_cleanup.csv" \
  --input ResidualFnCellConfidentMax2="$OUT/teacher_adjacent_dynamic_confidence_residual_fn_cell_confident_max2.csv" \
  --input ModerateIouCellTargetExtensionMax3="$OUT/teacher_adjacent_moderate_iou_cell_target_extension_max3.csv" \
  --output "$OUT/moderate_iou_teacher_fn_candidate_comparison.md" \
  --format markdown \
  --highlight-best \
  --include-best-summary \
  --include-reference-gap-summary \
  --reference-approach Track2p \
  --metric-output "$OUT/moderate_iou_teacher_fn_candidate_metrics.csv"

"$PY" -m bayescatrack benchmark compare \
  --input Track2p="$OUT/track2p_baseline.csv" \
  --input Track2pPolicyD12="$OUT/track2p_policy_d12.csv" \
  --input ComponentCleanup="$OUT/track2p_policy_component_cleanup.csv" \
  --input ResidualFnCellConfidentMax2="$OUT/teacher_adjacent_dynamic_confidence_residual_fn_cell_confident_max2.csv" \
  --input ModerateIouCellTargetExtensionMax3="$OUT/teacher_adjacent_moderate_iou_cell_target_extension_max3.csv" \
  --output "$OUT/moderate_iou_teacher_fn_candidate_comparison.csv" \
  --format csv

git status --short | tee "$OUT/git_status_after.txt" || true
echo 0 > "$OUT/exit_status.txt"
echo "completed_at=$(date --iso-8601=seconds)"
echo "$OUT"
