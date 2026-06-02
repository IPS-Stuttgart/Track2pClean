#!/usr/bin/env bash
set -euo pipefail

# Run the narrow Track2p-teacher seed-source repair candidates that remain
# plausible after the ComponentCleanup residual audit.
#
# Rationale:
#   - ComponentCleanup is the current frozen lead row for complete-track F1.
#   - Gap rescue and strict gated gap rescue did not improve official scores.
#   - Residual auditing points instead at Track2p-supported adjacent FNs and
#     missing seed-session ROI cases.
#
# This script keeps ComponentCleanup fixed and evaluates only label-free,
# Track2p-teacher adjacent rescue variants that spend at most two edits per
# subject on high-confidence seed-source repairs.  These rows are intentionally
# Track2p-teacher hybrid ablations, not independent BayesCaTrack trackers.
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
OUT=${OUT:-"$REPO/results/seed_source_teacher_rescue_candidates_$(date +%Y%m%d_%H%M%S)"}

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

# Preset row: macro configuration for the missing seed-session ROI residual
# bucket.  It enables seed-source backfill without broad source insertion and
# uses dynamic seed confidence ordering to spend the edit cap on seed evidence.
run_teacher_rescue teacher_adjacent_missing_seed_high_confidence \
  --teacher-repair-preset missing-seed-high-confidence

# Direct action-filter row: restrict teacher rescue to seed-source backfills.
# This is the narrowest Track2p-supported probe of the missing-seed bucket.
run_teacher_rescue teacher_adjacent_seed_source_only_high_confidence_max2 \
  --no-allow-completing-rescue \
  --allow-source-backfill \
  --allow-seed-source-backfill \
  --allow-completing-seed-source-backfill \
  --allow-fragment-merges \
  --teacher-edge-order dynamic-seed-confidence \
  --teacher-action-filter seed-source-backfill \
  --teacher-feature-preset high-confidence \
  --teacher-min-cell-probability 0.60 \
  --max-applied-edits 2

# Same structural hypothesis with the stricter cell-confidence preset.  This
# guards against recovering seed-source observations through low-cell endpoints.
run_teacher_rescue teacher_adjacent_seed_source_only_cell_confident_max2 \
  --no-allow-completing-rescue \
  --allow-source-backfill \
  --allow-seed-source-backfill \
  --allow-completing-seed-source-backfill \
  --allow-fragment-merges \
  --teacher-edge-order dynamic-seed-confidence \
  --teacher-action-filter seed-source-backfill \
  --teacher-feature-preset cell-confident \
  --max-applied-edits 2

"$PY" -m bayescatrack benchmark compare \
  --input Track2p="$OUT/track2p_baseline.csv" \
  --input Track2pPolicyD12="$OUT/track2p_policy_d12.csv" \
  --input ComponentCleanup="$OUT/track2p_policy_component_cleanup.csv" \
  --input TeacherAdjacentMissingSeedHighConfidence="$OUT/teacher_adjacent_missing_seed_high_confidence.csv" \
  --input TeacherAdjacentSeedSourceOnlyHighConfidenceMax2="$OUT/teacher_adjacent_seed_source_only_high_confidence_max2.csv" \
  --input TeacherAdjacentSeedSourceOnlyCellConfidentMax2="$OUT/teacher_adjacent_seed_source_only_cell_confident_max2.csv" \
  --output "$OUT/seed_source_teacher_rescue_candidates_comparison.md" \
  --format markdown \
  --highlight-best \
  --include-best-summary \
  --include-reference-gap-summary \
  --reference-approach Track2p \
  --metric-output "$OUT/seed_source_teacher_rescue_candidates_metrics.csv"

"$PY" -m bayescatrack benchmark compare \
  --input Track2p="$OUT/track2p_baseline.csv" \
  --input Track2pPolicyD12="$OUT/track2p_policy_d12.csv" \
  --input ComponentCleanup="$OUT/track2p_policy_component_cleanup.csv" \
  --input TeacherAdjacentMissingSeedHighConfidence="$OUT/teacher_adjacent_missing_seed_high_confidence.csv" \
  --input TeacherAdjacentSeedSourceOnlyHighConfidenceMax2="$OUT/teacher_adjacent_seed_source_only_high_confidence_max2.csv" \
  --input TeacherAdjacentSeedSourceOnlyCellConfidentMax2="$OUT/teacher_adjacent_seed_source_only_cell_confident_max2.csv" \
  --output "$OUT/seed_source_teacher_rescue_candidates_comparison.csv" \
  --format csv

git status --short | tee "$OUT/git_status_after.txt" || true
echo 0 > "$OUT/exit_status.txt"
echo "completed_at=$(date --iso-8601=seconds)"
echo "$OUT"
