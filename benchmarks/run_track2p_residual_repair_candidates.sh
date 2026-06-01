#!/usr/bin/env bash
set -euo pipefail

# Run the narrow residual-repair candidates that remain plausible after the
# ComponentCleanup residual audit.  The script is intentionally result-oriented:
# it keeps the frozen Track2pPolicy component-cleanup row fixed, then evaluates
# Track2p-teacher adjacent rescue variants that target residual adjacent false
# negatives and missing seed-session source observations.
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
OUT=${OUT:-"$REPO/results/residual_repair_candidates_$(date +%Y%m%d_%H%M%S)"}

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

run_teacher_veto() {
  local label=$1
  shift
  "$PY" -m bayescatrack benchmark track2p-policy-teacher-veto-cleanup \
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

# Audit the exact Track2p-supported adjacent FNs first.  This does not change
# predictions; it identifies whether any teacher-supported adjacent repairs have
# positive official what-if deltas and no duplicate/complete-TP side effects.
"$PY" -m bayescatrack benchmark track2p-policy-teacher-fn-audit \
  "${COMMON[@]}" \
  "${POLICY[@]}" \
  "${CLEANUP[@]}" \
  --feature-mode registered-subset \
  --output "$OUT/teacher_fn_edges.csv" \
  --summary-output "$OUT/teacher_fn_summary.csv" \
  --format csv

# Varying the seed session diagnoses the other residual bucket: complete-track
# FNs caused by the seed-anchored evaluation protocol rather than local evidence.
"$PY" -m bayescatrack benchmark track2p-policy-seed-sensitivity-audit \
  "${COMMON[@]}" \
  "${POLICY[@]}" \
  "${CLEANUP[@]}" \
  --seed-sessions all \
  --output "$OUT/seed_sensitivity_summary.csv" \
  --track-output "$OUT/seed_sensitivity_tracks.csv" \
  --format csv \
  --track-format csv

# Candidate rows.  These are intentionally narrow Track2p-teacher hybrid
# ablations; they do not use manual-GT labels to choose edges.  The opt-in rows
# test the two residual hypotheses: missing seed-source observations and row
# completion/fragment stitching.
run_teacher_rescue teacher_adjacent_default \
  --no-allow-completing-rescue \
  --allow-source-backfill \
  --no-allow-seed-source-backfill \
  --allow-fragment-merges

run_teacher_rescue teacher_adjacent_dynamic_structural \
  --no-allow-completing-rescue \
  --allow-source-backfill \
  --no-allow-seed-source-backfill \
  --allow-fragment-merges \
  --teacher-edge-order dynamic-structural

run_teacher_rescue teacher_adjacent_confidence \
  --no-allow-completing-rescue \
  --allow-source-backfill \
  --no-allow-seed-source-backfill \
  --allow-fragment-merges \
  --teacher-edge-order confidence

run_teacher_rescue teacher_adjacent_seed_source \
  --no-allow-completing-rescue \
  --allow-source-backfill \
  --allow-seed-source-backfill \
  --allow-fragment-merges

run_teacher_rescue teacher_adjacent_dynamic_confidence \
  --no-allow-completing-rescue \
  --allow-source-backfill \
  --no-allow-seed-source-backfill \
  --allow-fragment-merges \
  --teacher-edge-order dynamic-confidence

# Feature-gated rows: use Track2p teacher edges as a candidate generator, but
# require local registration/shape evidence before an edit may claim a slot.
run_teacher_rescue teacher_adjacent_feature_gated_dynamic_confidence \
  --no-allow-completing-rescue \
  --allow-source-backfill \
  --no-allow-seed-source-backfill \
  --allow-fragment-merges \
  --teacher-edge-order dynamic-confidence \
  --teacher-min-registered-iou 0.10 \
  --teacher-max-centroid-distance 6.0 \
  --teacher-min-area-ratio 0.45 \
  --min-component-observations 2

run_teacher_rescue teacher_adjacent_feature_gated_dynamic_confidence_max1 \
  --no-allow-completing-rescue \
  --allow-source-backfill \
  --no-allow-seed-source-backfill \
  --allow-fragment-merges \
  --teacher-edge-order dynamic-confidence \
  --teacher-min-registered-iou 0.10 \
  --teacher-max-centroid-distance 6.0 \
  --teacher-min-area-ratio 0.45 \
  --min-component-observations 2 \
  --max-applied-edits 1

# Stricter label-free support gate: only rescue components that already contain
# at least two observations.
run_teacher_rescue teacher_adjacent_supported \
  --no-allow-completing-rescue \
  --allow-source-backfill \
  --no-allow-seed-source-backfill \
  --allow-fragment-merges \
  --min-component-observations 2

# Few-edit rows: residual audits suggest the next useful repair is likely one or
# two high-priority adjacent teacher edits, not admitting the full teacher edge
# tail. These rows keep the same label-free dynamic-confidence ordering but cap
# accepted edits per subject.
run_teacher_rescue teacher_adjacent_dynamic_confidence_max1 \
  --no-allow-completing-rescue \
  --allow-source-backfill \
  --no-allow-seed-source-backfill \
  --allow-fragment-merges \
  --teacher-edge-order dynamic-confidence \
  --max-applied-edits 1

run_teacher_rescue teacher_adjacent_dynamic_confidence_max2 \
  --no-allow-completing-rescue \
  --allow-source-backfill \
  --no-allow-seed-source-backfill \
  --allow-fragment-merges \
  --teacher-edge-order dynamic-confidence \
  --max-applied-edits 2

# Feature-gated few-edit rows.  These target the same Track2p-supported adjacent
# FN and missing-seed buckets, but avoid admitting the full Track2p-teacher tail.
run_teacher_rescue teacher_adjacent_dynamic_confidence_local_support_max2 \
  --no-allow-completing-rescue \
  --allow-source-backfill \
  --no-allow-seed-source-backfill \
  --allow-fragment-merges \
  --teacher-edge-order dynamic-confidence \
  --teacher-feature-preset local-support \
  --max-applied-edits 2

run_teacher_rescue teacher_adjacent_dynamic_confidence_high_confidence_seed_source_max2 \
  --no-allow-completing-rescue \
  --allow-source-backfill \
  --allow-seed-source-backfill \
  --allow-fragment-merges \
  --teacher-edge-order dynamic-confidence \
  --teacher-feature-preset high-confidence \
  --max-applied-edits 2

run_teacher_rescue teacher_adjacent_completing \
  --allow-completing-rescue \
  --allow-source-backfill \
  --no-allow-seed-source-backfill \
  --allow-fragment-merges

run_teacher_rescue teacher_adjacent_completing_seed_source \
  --allow-completing-rescue \
  --allow-source-backfill \
  --allow-seed-source-backfill \
  --allow-fragment-merges

# Opt-in candidate: recompute the label-free structural priority after each
# accepted teacher edit while allowing completion.
run_teacher_rescue teacher_adjacent_dynamic_completing_seed_source \
  --allow-completing-rescue \
  --allow-source-backfill \
  --allow-seed-source-backfill \
  --allow-fragment-merges \
  --teacher-edge-order dynamic-structural

# Exact completion row: only allow teacher edits to complete a row when the
# completed row is itself present as a complete Track2p teacher row.
run_teacher_rescue teacher_adjacent_teacher_complete_row \
  --no-allow-completing-rescue \
  --allow-teacher-complete-row-rescue \
  --allow-source-backfill \
  --allow-seed-source-backfill \
  --allow-fragment-merges \
  --teacher-edge-order dynamic-confidence

# Conservative teacher-veto rows target the other residual family: Bayes-only
# adjacent false continuations. The geometric row only tests vetoes that are
# both absent from Track2p and locally weak by centroid/area evidence.
run_teacher_veto teacher_veto_default \
  --no-allow-complete-track-veto \
  --max-threshold-margin 0.10 \
  --max-competition-margin 0.20 \
  --min-veto-fragment-observations 2

run_teacher_veto teacher_veto_geometric_max1 \
  --no-allow-complete-track-veto \
  --max-threshold-margin 0.10 \
  --max-competition-margin 0.20 \
  --min-centroid-distance 3.0 \
  --max-area-ratio 0.65 \
  --min-veto-fragment-observations 2 \
  --max-applied-vetoes 1

"$PY" -m bayescatrack benchmark compare \
  --input Track2p="$OUT/track2p_baseline.csv" \
  --input Track2pPolicyD12="$OUT/track2p_policy_d12.csv" \
  --input ComponentCleanup="$OUT/track2p_policy_component_cleanup.csv" \
  --input TeacherAdjacentDefault="$OUT/teacher_adjacent_default.csv" \
  --input TeacherAdjacentDynamicStructural="$OUT/teacher_adjacent_dynamic_structural.csv" \
  --input TeacherAdjacentConfidence="$OUT/teacher_adjacent_confidence.csv" \
  --input TeacherAdjacentSeedSource="$OUT/teacher_adjacent_seed_source.csv" \
  --input TeacherAdjacentDynamicConfidence="$OUT/teacher_adjacent_dynamic_confidence.csv" \
  --input TeacherAdjacentFeatureGatedDynamicConfidence="$OUT/teacher_adjacent_feature_gated_dynamic_confidence.csv" \
  --input TeacherAdjacentFeatureGatedDynamicConfidenceMax1="$OUT/teacher_adjacent_feature_gated_dynamic_confidence_max1.csv" \
  --input TeacherAdjacentSupported="$OUT/teacher_adjacent_supported.csv" \
  --input TeacherAdjacentDynamicConfidenceMax1="$OUT/teacher_adjacent_dynamic_confidence_max1.csv" \
  --input TeacherAdjacentDynamicConfidenceMax2="$OUT/teacher_adjacent_dynamic_confidence_max2.csv" \
  --input TeacherAdjacentDynamicConfidenceLocalSupportMax2="$OUT/teacher_adjacent_dynamic_confidence_local_support_max2.csv" \
  --input TeacherAdjacentDynamicConfidenceHighConfidenceSeedSourceMax2="$OUT/teacher_adjacent_dynamic_confidence_high_confidence_seed_source_max2.csv" \
  --input TeacherAdjacentCompleting="$OUT/teacher_adjacent_completing.csv" \
  --input TeacherAdjacentCompletingSeedSource="$OUT/teacher_adjacent_completing_seed_source.csv" \
  --input TeacherAdjacentDynamicCompletingSeedSource="$OUT/teacher_adjacent_dynamic_completing_seed_source.csv" \
  --input TeacherAdjacentTeacherCompleteRow="$OUT/teacher_adjacent_teacher_complete_row.csv" \
  --input TeacherVetoDefault="$OUT/teacher_veto_default.csv" \
  --input TeacherVetoGeometricMax1="$OUT/teacher_veto_geometric_max1.csv" \
  --output "$OUT/residual_repair_candidates_comparison.md" \
  --format markdown \
  --highlight-best \
  --include-best-summary \
  --include-reference-gap-summary \
  --reference-approach Track2p \
  --metric-output "$OUT/residual_repair_candidates_metrics.csv"

"$PY" -m bayescatrack benchmark compare \
  --input Track2p="$OUT/track2p_baseline.csv" \
  --input Track2pPolicyD12="$OUT/track2p_policy_d12.csv" \
  --input ComponentCleanup="$OUT/track2p_policy_component_cleanup.csv" \
  --input TeacherAdjacentDefault="$OUT/teacher_adjacent_default.csv" \
  --input TeacherAdjacentDynamicStructural="$OUT/teacher_adjacent_dynamic_structural.csv" \
  --input TeacherAdjacentConfidence="$OUT/teacher_adjacent_confidence.csv" \
  --input TeacherAdjacentSeedSource="$OUT/teacher_adjacent_seed_source.csv" \
  --input TeacherAdjacentDynamicConfidence="$OUT/teacher_adjacent_dynamic_confidence.csv" \
  --input TeacherAdjacentFeatureGatedDynamicConfidence="$OUT/teacher_adjacent_feature_gated_dynamic_confidence.csv" \
  --input TeacherAdjacentFeatureGatedDynamicConfidenceMax1="$OUT/teacher_adjacent_feature_gated_dynamic_confidence_max1.csv" \
  --input TeacherAdjacentSupported="$OUT/teacher_adjacent_supported.csv" \
  --input TeacherAdjacentDynamicConfidenceMax1="$OUT/teacher_adjacent_dynamic_confidence_max1.csv" \
  --input TeacherAdjacentDynamicConfidenceMax2="$OUT/teacher_adjacent_dynamic_confidence_max2.csv" \
  --input TeacherAdjacentDynamicConfidenceLocalSupportMax2="$OUT/teacher_adjacent_dynamic_confidence_local_support_max2.csv" \
  --input TeacherAdjacentDynamicConfidenceHighConfidenceSeedSourceMax2="$OUT/teacher_adjacent_dynamic_confidence_high_confidence_seed_source_max2.csv" \
  --input TeacherAdjacentCompleting="$OUT/teacher_adjacent_completing.csv" \
  --input TeacherAdjacentCompletingSeedSource="$OUT/teacher_adjacent_completing_seed_source.csv" \
  --input TeacherAdjacentDynamicCompletingSeedSource="$OUT/teacher_adjacent_dynamic_completing_seed_source.csv" \
  --input TeacherAdjacentTeacherCompleteRow="$OUT/teacher_adjacent_teacher_complete_row.csv" \
  --input TeacherVetoDefault="$OUT/teacher_veto_default.csv" \
  --input TeacherVetoGeometricMax1="$OUT/teacher_veto_geometric_max1.csv" \
  --output "$OUT/residual_repair_candidates_comparison.csv" \
  --format csv

git status --short | tee "$OUT/git_status_after.txt" || true
echo 0 > "$OUT/exit_status.txt"
echo "completed_at=$(date --iso-8601=seconds)"
echo "$OUT"
