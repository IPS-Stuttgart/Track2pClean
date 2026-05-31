#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-$(pwd)}
DATA=${DATA:?Set DATA to the Track2p-style data root or resolved lightweight view}
REF=${REF:-$DATA}
PY=${PY:-python}
OUT=${OUT:-"$REPO/results/dynamic_confidence_completion_rescue_$(date +%Y%m%d_%H%M%S)"}

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

COMMON=(--data "$DATA" --reference "$REF" --reference-kind manual-gt --input-format suite2p --plane plane0 --no-include-behavior)
POLICY=(--threshold-method min --transform-type affine --iou-distance-threshold 12 --cell-probability-threshold 0.5)
CLEANUP=(--split-risk-threshold 1.5 --min-side-observations 2)

"$PY" -m bayescatrack benchmark track2p "${COMMON[@]}" --method track2p-baseline --output "$OUT/track2p_baseline.csv" --format csv
"$PY" -m bayescatrack benchmark track2p-policy "${COMMON[@]}" "${POLICY[@]}" --output "$OUT/track2p_policy_d12.csv" --format csv
"$PY" -m bayescatrack benchmark track2p-policy-component-audit "${COMMON[@]}" "${POLICY[@]}" --apply-splits "${CLEANUP[@]}" --output "$OUT/track2p_policy_component_cleanup.csv" --format csv --component-output "$OUT/track2p_policy_component_cleanup_components.csv" --component-format csv
"$PY" -m bayescatrack.experiments.track2p_policy_dynamic_completion_rescue "${COMMON[@]}" "${POLICY[@]}" "${CLEANUP[@]}" --output "$OUT/dynamic_confidence_completion_rescue.csv" --format csv --diagnostics-output "$OUT/dynamic_confidence_completion_rescue_edges.csv" --diagnostics-format csv

"$PY" -m bayescatrack benchmark compare \
  --input Track2p="$OUT/track2p_baseline.csv" \
  --input Track2pPolicyD12="$OUT/track2p_policy_d12.csv" \
  --input ComponentCleanup="$OUT/track2p_policy_component_cleanup.csv" \
  --input DynamicConfidenceCompletionRescue="$OUT/dynamic_confidence_completion_rescue.csv" \
  --output "$OUT/dynamic_confidence_completion_rescue_comparison.md" \
  --format markdown \
  --highlight-best \
  --include-best-summary \
  --include-reference-gap-summary \
  --reference-approach Track2p \
  --metric-output "$OUT/dynamic_confidence_completion_rescue_metrics.csv"

"$PY" -m bayescatrack benchmark compare \
  --input Track2p="$OUT/track2p_baseline.csv" \
  --input Track2pPolicyD12="$OUT/track2p_policy_d12.csv" \
  --input ComponentCleanup="$OUT/track2p_policy_component_cleanup.csv" \
  --input DynamicConfidenceCompletionRescue="$OUT/dynamic_confidence_completion_rescue.csv" \
  --output "$OUT/dynamic_confidence_completion_rescue_comparison.csv" \
  --format csv

git status --short | tee "$OUT/git_status_after.txt" || true
echo 0 > "$OUT/exit_status.txt"
echo "completed_at=$(date --iso-8601=seconds)"
echo "$OUT"
