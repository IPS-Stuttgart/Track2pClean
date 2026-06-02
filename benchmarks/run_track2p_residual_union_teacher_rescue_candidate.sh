#!/usr/bin/env bash
set -euo pipefail

# Run the narrow residual-union teacher-rescue candidate that combines the two
# residual buckets that remain plausible after ComponentCleanup:
#
#   1. Track2p-supported adjacent false negatives via target extension.
#   2. Missing seed-session source observations via seed-source backfill.
#
# The candidate deliberately avoids broad source backfill, fragment merges, and
# arbitrary complete-row target insertions.  It uses Track2p only as a teacher
# candidate generator, then requires the residual-FN cell-confident local feature
# gate and a tiny per-subject edit budget.  Manual-GT labels are used only by the
# benchmark scorer, not by candidate selection.
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
OUT=${OUT:-"$REPO/results/residual_union_teacher_rescue_$(date +%Y%m%d_%H%M%S)"}

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
UNION_RESCUE=(
  --no-allow-completing-rescue
  --no-allow-source-backfill
  --allow-seed-source-backfill
  --allow-completing-seed-source-backfill
  --no-allow-fragment-merges
  --teacher-edge-order dynamic-seed-confidence
  --teacher-action-filter all
  --teacher-feature-preset residual-fn-cell-confident
  --min-component-observations 2
)

run_teacher_union() {
  local label=$1
  local edit_cap=$2
  "$PY" -m bayescatrack benchmark track2p-policy-teacher-adjacent-rescue \
    "${COMMON[@]}" \
    "${POLICY[@]}" \
    "${CLEANUP[@]}" \
    "${UNION_RESCUE[@]}" \
    --max-applied-edits "$edit_cap" \
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

run_teacher_union teacher_residual_union_max1 1
run_teacher_union teacher_residual_union_max2 2
run_teacher_union teacher_residual_union_max3 3

"$PY" -m bayescatrack benchmark compare \
  --input Track2p="$OUT/track2p_baseline.csv" \
  --input Track2pPolicyD12="$OUT/track2p_policy_d12.csv" \
  --input ComponentCleanup="$OUT/track2p_policy_component_cleanup.csv" \
  --input TeacherResidualUnionMax1="$OUT/teacher_residual_union_max1.csv" \
  --input TeacherResidualUnionMax2="$OUT/teacher_residual_union_max2.csv" \
  --input TeacherResidualUnionMax3="$OUT/teacher_residual_union_max3.csv" \
  --output "$OUT/residual_union_teacher_rescue_comparison.md" \
  --format markdown \
  --highlight-best \
  --include-best-summary \
  --include-reference-gap-summary \
  --reference-approach Track2p \
  --metric-output "$OUT/residual_union_teacher_rescue_metrics.csv"

"$PY" -m bayescatrack benchmark compare \
  --input Track2p="$OUT/track2p_baseline.csv" \
  --input Track2pPolicyD12="$OUT/track2p_policy_d12.csv" \
  --input ComponentCleanup="$OUT/track2p_policy_component_cleanup.csv" \
  --input TeacherResidualUnionMax1="$OUT/teacher_residual_union_max1.csv" \
  --input TeacherResidualUnionMax2="$OUT/teacher_residual_union_max2.csv" \
  --input TeacherResidualUnionMax3="$OUT/teacher_residual_union_max3.csv" \
  --output "$OUT/residual_union_teacher_rescue_comparison.csv" \
  --format csv

git status --short | tee "$OUT/git_status_after.txt" || true
echo 0 > "$OUT/exit_status.txt"
echo "completed_at=$(date --iso-8601=seconds)"
echo "$OUT"
