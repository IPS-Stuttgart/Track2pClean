# FullMHT Identity-History Validation, 2026-06-26

This note defines the next paper-facing FullMHT candidate bundle.  The goal is to
test whether FullMHT has become more than cleanup: a complete-history identity
tracker whose beam search can beat the matching greedy history on real benchmark
data.

## Candidate Row

The frozen candidate is:

```text
FullMHTIdentityHistory
```

It combines these label-free model layers inside the scan-assignment MHT score:

```text
calibrated association likelihood
+ strong Track2p proposal prior
+ calibrated Track2p prior-edge survival likelihood
+ calibrated no-prior continuation likelihood
+ growth-history prediction penalty
```

The matching greedy ablation is:

```text
FullMHTGreedyIdentityHistory
```

It uses the same scan candidates and scoring terms, but sets `beam_width = 1` and
turns off identity-diverse beam retention.  Promotion requires the full beam to
beat this greedy row on complete-track F1 without pairwise-F1 loss.

## Frozen Artifacts

| artifact | purpose |
| --- | --- |
| `benchmarks/full_mht_identity_history_candidate_manifest.json` | canonical comparison against Track2p, prior-only FullMHT, prior-survival, no-prior continuation, and greedy identity-history |
| `benchmarks/full_mht_identity_history_sensitivity_manifest.json` | immediate-neighborhood sensitivity around survival weight, no-prior continuation weight, and growth-history weight |
| `full_mht_identity_history_decision.py` | interprets the canonical comparison table |
| `full_mht_identity_history_promotion_gate.py` | combines canonical decision, sensitivity, and label-free exposure audit |
| `track2p_policy_full_mht_exposure_audit.py` | runs all Track2p-style subjects without loading references or audit labels |

## Decision Rule

Promote only if all of these are true:

- `FullMHTIdentityHistory` has complete-track advantage over `FullMHTGreedyIdentityHistory` with no pairwise-F1 loss.
- It does not fall below `Track2p`, `FullMHTPrior2`, `FullMHTPriorSurvival`, or `FullMHTNoPriorContinuation100` on the required micro metrics.
- The sensitivity manifest reports `stable_plateau`.
- The exposure audit reports `bounded_exposure`.
- Prior-survival, no-prior continuation, and growth-history signals are active but not broad.
- The no-GT leakage regression passes.

If any gate fails, keep the row exploratory.  A tie against greedy means the
benchmark still does not prove that MHT history search, rather than local scoring,
is responsible for the result.

## Server Bundle

Run this on the Python 3.12 benchmark environment:

```bash
set -euo pipefail

REPO=/home/florianpfaff/codex-runs/BayesCaTrack
PY="$REPO/.venv312/bin/python"
DATA="$REPO/results/policy_dp/data_lightweight"

cd "$REPO"
git fetch origin
git checkout codex/full-mht-prototype
git reset --hard origin/codex/full-mht-prototype
export PYTHONPATH="$REPO/src"

"$PY" -m pytest -q \
  tests/test_full_mht_identity_history_candidate_manifest.py \
  tests/test_full_mht_identity_history_sensitivity_manifest.py \
  tests/test_full_mht_identity_history_decision.py \
  tests/test_full_mht_identity_history_promotion_gate.py \
  tests/test_full_mht_no_gt_leakage.py \
  tests/test_full_mht_exposure_audit.py

IDH="$REPO/results/full_mht_identity_history_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$IDH"
"$PY" -m bayescatrack benchmark suite \
  benchmarks/full_mht_identity_history_candidate_manifest.json \
  --output-dir "$IDH" \
  --summary-format table

"$PY" -m bayescatrack.experiments.full_mht_identity_history_decision \
  "$IDH/full_mht_identity_history/full_mht_identity_history_comparison.csv" \
  --output "$IDH/full_mht_identity_history_decision.md"

SENS="$REPO/results/full_mht_identity_history_sensitivity_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SENS"
"$PY" -m bayescatrack benchmark suite \
  benchmarks/full_mht_identity_history_sensitivity_manifest.json \
  --output-dir "$SENS" \
  --summary-format table

EXPOSURE="$REPO/results/full_mht_identity_history_exposure_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EXPOSURE"
"$PY" -m bayescatrack.experiments.track2p_policy_full_mht_exposure_audit \
  --data "$DATA" \
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
  --association-score-mode calibrated-likelihood \
  --association-likelihood-weight 1.0 \
  --association-likelihood-clip 4.0 \
  --track2p-prior-weight 12.0 \
  --track2p-non-prior-penalty 2.0 \
  --track2p-prior-switch-penalty 8.0 \
  --track2p-no-prior-successor-penalty 0.0 \
  --track2p-prior-miss-penalty 4.0 \
  --track2p-prior-survival-weight 1.0 \
  --track2p-prior-survival-min-examples-per-class 2 \
  --track2p-prior-survival-score-clip 8.0 \
  --no-prior-continuation-likelihood-weight 1.0 \
  --no-prior-continuation-min-examples-per-class 2 \
  --no-prior-continuation-score-clip 8.0 \
  --growth-history-prediction-weight 0.5 \
  --growth-history-prediction-scale 1.0 \
  --growth-history-prediction-clip 8.0 \
  --growth-history-prediction-min-edges 1 \
  --output "$EXPOSURE/full_mht_identity_history_exposure.csv" \
  --format csv \
  --progress

"$PY" -m bayescatrack.experiments.full_mht_identity_history_promotion_gate \
  "$IDH/full_mht_identity_history/full_mht_identity_history_comparison.csv" \
  "$SENS/full_mht_identity_history_sensitivity/full_mht_identity_history_sensitivity_comparison.csv" \
  "$EXPOSURE/full_mht_identity_history_exposure.csv" \
  --output "$EXPOSURE/full_mht_identity_history_promotion_gate.md"
```

## How To Interpret Outcomes

| gate result | interpretation |
| --- | --- |
| `promotable_after_review` | strong candidate for the paper method row, after recording exact directories and metric tables |
| `not_promotable_manifest` | no real-data proof that MHT history search beats greedy local selection |
| `not_promotable_sensitivity` | likely knife-edge or single-setting result |
| `not_promotable_broad_exposure` | model layer fires too broadly on label-free subjects |
| `incomplete` | rerun the missing manifest, sensitivity, exposure, or no-GT test artifact |

The branch should not claim an original FullMHT method row until this bundle has
been run and recorded.
