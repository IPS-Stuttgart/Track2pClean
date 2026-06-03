#!/usr/bin/env bash
set -euo pipefail

# Focused follow-up for the residual official-error audit.
#
# The frozen Track2pPolicy-family lead already uses ComponentCleanup (and, in the
# broader next-steps manifest, CoherenceSuffixStitch).  The residual audit showed
# two remaining potentially actionable buckets: Track2p-supported adjacent FNs and
# missing seed-session source observations.  This runner evaluates the existing
# action-specific residual-union teacher-rescue preset with a tiny edit budget, so
# target-extension candidates and seed-source backfills use different label-free
# feature gates instead of sharing one permissive gate.
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
OUT=${OUT:-"$REPO/results/residual_union_action_specific_$(date +%Y%m%d_%H%M%S)"}

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

# Baseline residual-union preset: shared cell-confident gate, edit budget from the
# preset.  Kept here as the direct comparator to the action-specific split gates.
run_teacher_rescue residual_union_cell_confident \
  --no-allow-completing-rescue \
  --allow-source-backfill \
  --allow-seed-source-backfill \
  --allow-completing-seed-source-backfill \
  --no-allow-fragment-merges \
  --teacher-repair-preset residual-union-cell-confident

# Action-specific residual union: target extensions get the moderate-IoU/cell gate
# while seed-source backfills get the seed-source cell-confident gate.  We test
# tiny edit budgets because the residual audit suggests any useful repair is likely
# one or two edits, not a broad Track2p-teacher edge tail.
run_teacher_rescue residual_union_action_specific_max1 \
  --no-allow-completing-rescue \
  --no-allow-source-backfill \
  --allow-seed-source-backfill \
  --allow-completing-seed-source-backfill \
  --no-allow-fragment-merges \
  --teacher-repair-preset residual-union-action-specific \
  --max-applied-edits 1

run_teacher_rescue residual_union_action_specific_max2 \
  --no-allow-completing-rescue \
  --no-allow-source-backfill \
  --allow-seed-source-backfill \
  --allow-completing-seed-source-backfill \
  --no-allow-fragment-merges \
  --teacher-repair-preset residual-union-action-specific \
  --max-applied-edits 2

run_teacher_rescue residual_union_action_specific_default \
  --no-allow-completing-rescue \
  --no-allow-source-backfill \
  --allow-seed-source-backfill \
  --allow-completing-seed-source-backfill \
  --no-allow-fragment-merges \
  --teacher-repair-preset residual-union-action-specific

"$PY" -m bayescatrack benchmark compare \
  --input Track2p="$OUT/track2p_baseline.csv" \
  --input Track2pPolicyD12="$OUT/track2p_policy_d12.csv" \
  --input ComponentCleanup="$OUT/track2p_policy_component_cleanup.csv" \
  --input ResidualUnionCellConfident="$OUT/residual_union_cell_confident.csv" \
  --input ResidualUnionActionSpecificMax1="$OUT/residual_union_action_specific_max1.csv" \
  --input ResidualUnionActionSpecificMax2="$OUT/residual_union_action_specific_max2.csv" \
  --input ResidualUnionActionSpecificDefault="$OUT/residual_union_action_specific_default.csv" \
  --output "$OUT/residual_union_action_specific_comparison.md" \
  --format markdown \
  --highlight-best \
  --include-best-summary \
  --include-reference-gap-summary \
  --reference-approach Track2p \
  --metric-output "$OUT/residual_union_action_specific_metrics.csv"

"$PY" -m bayescatrack benchmark compare \
  --input Track2p="$OUT/track2p_baseline.csv" \
  --input Track2pPolicyD12="$OUT/track2p_policy_d12.csv" \
  --input ComponentCleanup="$OUT/track2p_policy_component_cleanup.csv" \
  --input ResidualUnionCellConfident="$OUT/residual_union_cell_confident.csv" \
  --input ResidualUnionActionSpecificMax1="$OUT/residual_union_action_specific_max1.csv" \
  --input ResidualUnionActionSpecificMax2="$OUT/residual_union_action_specific_max2.csv" \
  --input ResidualUnionActionSpecificDefault="$OUT/residual_union_action_specific_default.csv" \
  --output "$OUT/residual_union_action_specific_comparison.csv" \
  --format csv

git status --short | tee "$OUT/git_status_after.txt" || true
echo 0 > "$OUT/exit_status.txt"
echo "completed_at=$(date --iso-8601=seconds)"
echo "$OUT"
