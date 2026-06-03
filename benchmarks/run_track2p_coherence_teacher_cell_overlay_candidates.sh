#!/usr/bin/env bash
set -euo pipefail

# Focused follow-up for the current Track2p-policy-family lead.
#
# The independent/non-teacher row is CoherenceSuffixStitch. This script keeps
# that row fixed, then evaluates small Track2p-teacher overlay variants that use
# endpoint cell probability as a label-free tie-breaker/gate.
#
# Required:
#   DATA=/path/to/track2p-style/data-or-resolved-lightweight-view
#
# Optional:
#   REF=/path/to/manual-gt-reference-root
#   REPO=/path/to/BayesCaTrack
#   OUT=/path/to/output-dir
#   PY=/path/to/python

REPO=${REPO:-$(pwd)}
DATA=${DATA:?Set DATA to the Track2p-style data root or resolved lightweight view}
REF=${REF:-$DATA}
PY=${PY:-python}
OUT=${OUT:-"$REPO/results/coherence_teacher_cell_overlay_$(date +%Y%m%d_%H%M%S)"}

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
SUFFIX=(
  --suffix-path-length 2
  --min-cell-probability 0.80
  --min-area-ratio 0.80
  --max-centroid-distance 6.0
  --min-shifted-iou 0.30
  --min-motion-consistency 0.50
  --min-shape-consistency 0.82
  --max-stitches-per-subject 1
  --edge-top-k 25
  --path-beam-width 100
)

run_suffix_teacher() {
  local label=$1
  shift
  "$PY" -m bayescatrack benchmark track2p-policy-coherence-suffix-teacher-rescue \
    "${COMMON[@]}" \
    "${POLICY[@]}" \
    "${CLEANUP[@]}" \
    "${SUFFIX[@]}" \
    "$@" \
    --output "$OUT/${label}.csv" \
    --teacher-output "$OUT/${label}_teacher_edges.csv" \
    --format csv
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

"$PY" -m bayescatrack benchmark track2p-policy-coherence-suffix-stitch \
  "${COMMON[@]}" \
  "${POLICY[@]}" \
  "${CLEANUP[@]}" \
  "${SUFFIX[@]}" \
  --output "$OUT/track2p_policy_coherence_suffix_stitch.csv" \
  --candidate-output "$OUT/track2p_policy_coherence_suffix_stitch_candidates.csv" \
  --format csv

run_suffix_teacher coherence_suffix_teacher_structural \
  --teacher-edge-order structural \
  --teacher-action-filter all \
  --teacher-feature-preset none \
  --max-applied-teacher-edits -1

run_suffix_teacher coherence_suffix_teacher_dynamic_cell_high_confidence_max1 \
  --teacher-edge-order dynamic-cell-confidence \
  --teacher-action-filter all \
  --teacher-feature-preset cell-high-confidence \
  --max-applied-teacher-edits 1

run_suffix_teacher coherence_suffix_teacher_dynamic_cell_high_confidence_max2 \
  --teacher-edge-order dynamic-cell-confidence \
  --teacher-action-filter all \
  --teacher-feature-preset cell-high-confidence \
  --max-applied-teacher-edits 2

run_suffix_teacher coherence_suffix_teacher_dynamic_cell_local_support_max2 \
  --teacher-edge-order dynamic-cell-confidence \
  --teacher-action-filter all \
  --teacher-feature-preset local-support \
  --max-applied-teacher-edits 2

"$PY" -m bayescatrack benchmark compare \
  --input Track2p="$OUT/track2p_baseline.csv" \
  --input Track2pPolicyD12="$OUT/track2p_policy_d12.csv" \
  --input ComponentCleanup="$OUT/track2p_policy_component_cleanup.csv" \
  --input CoherenceSuffixStitch="$OUT/track2p_policy_coherence_suffix_stitch.csv" \
  --input CoherenceSuffixTeacherStructural="$OUT/coherence_suffix_teacher_structural.csv" \
  --input CoherenceSuffixTeacherDynamicCellHighConfidenceMax1="$OUT/coherence_suffix_teacher_dynamic_cell_high_confidence_max1.csv" \
  --input CoherenceSuffixTeacherDynamicCellHighConfidenceMax2="$OUT/coherence_suffix_teacher_dynamic_cell_high_confidence_max2.csv" \
  --input CoherenceSuffixTeacherDynamicCellLocalSupportMax2="$OUT/coherence_suffix_teacher_dynamic_cell_local_support_max2.csv" \
  --output "$OUT/coherence_teacher_cell_overlay_comparison.md" \
  --format markdown \
  --highlight-best \
  --include-best-summary \
  --include-reference-gap-summary \
  --reference-approach Track2p \
  --metric-output "$OUT/coherence_teacher_cell_overlay_metrics.csv"

"$PY" -m bayescatrack benchmark compare \
  --input Track2p="$OUT/track2p_baseline.csv" \
  --input Track2pPolicyD12="$OUT/track2p_policy_d12.csv" \
  --input ComponentCleanup="$OUT/track2p_policy_component_cleanup.csv" \
  --input CoherenceSuffixStitch="$OUT/track2p_policy_coherence_suffix_stitch.csv" \
  --input CoherenceSuffixTeacherStructural="$OUT/coherence_suffix_teacher_structural.csv" \
  --input CoherenceSuffixTeacherDynamicCellHighConfidenceMax1="$OUT/coherence_suffix_teacher_dynamic_cell_high_confidence_max1.csv" \
  --input CoherenceSuffixTeacherDynamicCellHighConfidenceMax2="$OUT/coherence_suffix_teacher_dynamic_cell_high_confidence_max2.csv" \
  --input CoherenceSuffixTeacherDynamicCellLocalSupportMax2="$OUT/coherence_suffix_teacher_dynamic_cell_local_support_max2.csv" \
  --output "$OUT/coherence_teacher_cell_overlay_comparison.csv" \
  --format csv

git status --short | tee "$OUT/git_status_after.txt" || true
echo 0 > "$OUT/exit_status.txt"
echo "completed_at=$(date --iso-8601=seconds)"
echo "$OUT"
