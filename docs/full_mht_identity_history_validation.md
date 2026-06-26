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

## Constructed Conflict Witness

The branch also keeps a small label-free conflict witness in
`track2p-policy-full-mht-conflict-demo`.  It is not a benchmark result; it is a
mechanical proof-of-behavior for the method claim.  The demo constructs scan
assignment histories where the locally best edge leads to a later dead end, while
a slightly weaker edge preserves a complete identity once later evidence arrives.

The focused regression requires two invariants:

```text
full MHT beam > greedy beam-width-1 on the dead-end conflict
full MHT beam improves complete-track F1 while greedy remains pairwise-good
```

This witness does not promote the real-data row by itself.  It only protects the
claim that the MHT beam can be load-bearing when identity histories conflict.

## Complete-History Objective Probe

The identity-history candidate is not silently changed by this note.  A separate
frozen probe asks whether adding the terminal complete-history objective to the
same combined model helps:

```text
FullMHTIdentityHistoryCompletion025
FullMHTIdentityHistoryCompletion050
FullMHTIdentityHistoryCompletion100
```

These rows are identical to `FullMHTIdentityHistory` except for
`terminal_incomplete_history_weight`.  The terminal objective can enter the
paper-facing method only if at least two nearby weights improve complete-track F1
without pairwise-F1 regression.  A single winning weight is treated as exploratory.

## Frozen Artifacts

| artifact | purpose |
| --- | --- |
| `benchmarks/full_mht_identity_history_candidate_manifest.json` | canonical comparison against Track2p, prior-only FullMHT, prior-survival, no-prior continuation, and greedy identity-history |
| `benchmarks/full_mht_identity_history_sensitivity_manifest.json` | immediate-neighborhood sensitivity around survival weight, no-prior continuation weight, and growth-history weight |
| `benchmarks/full_mht_identity_history_completion_manifest.json` | complete-history terminal objective probe on top of the combined identity-history row |
| `track2p_policy_full_mht_conflict_demo.py` | constructed witness that full-history beam search can beat greedy local assignment in an identity-history conflict |
| `test_track2p_policy_full_mht_conflict_demo.py` | regression for the constructed MHT-vs-greedy conflict witness |
| `full_mht_identity_history_decision.py` | interprets the canonical comparison table |
| `full_mht_identity_history_promotion_gate.py` | combines canonical decision, sensitivity, and label-free exposure audit |
| `full_mht_terminal_completion_decision.py` | interprets the terminal-completion probe, with row-name overrides for identity-history rows |
| `track2p_policy_full_mht_exposure_audit.py` | runs all Track2p-style subjects without loading references or audit labels |

## Decision Rule

Promote `FullMHTIdentityHistory` only if all of these are true:

- `FullMHTIdentityHistory` has complete-track advantage over `FullMHTGreedyIdentityHistory` with no pairwise-F1 loss.
- It does not fall below `Track2p`, `FullMHTPrior2`, `FullMHTPriorSurvival`, or `FullMHTNoPriorContinuation100` on the required micro metrics.
- The constructed conflict witness regression passes.
- The sensitivity manifest reports `stable_plateau`.
- The exposure audit reports `bounded_exposure`.
- Prior-survival, no-prior continuation, and growth-history signals are active but not broad.
- The no-GT leakage regression passes.

Promote a terminal-completion variant only if the identity-history row itself
passes those gates and the completion probe reports `terminal_completion_stable_gain`.
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
  tests/test_full_mht_identity_history_completion_manifest.py \
  tests/test_full_mht_identity_history_decision.py \
  tests/test_full_mht_identity_history_promotion_gate.py \
  tests/test_full_mht_terminal_completion_decision.py \
  tests/test_track2p_policy_full_mht_conflict_demo.py \
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

COMP="$REPO/results/full_mht_identity_history_completion_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$COMP"
"$PY" -m bayescatrack benchmark suite \
  benchmarks/full_mht_identity_history_completion_manifest.json \
  --output-dir "$COMP" \
  --summary-format table

"$PY" -m bayescatrack.experiments.full_mht_terminal_completion_decision \
  "$COMP/full_mht_identity_history_completion/full_mht_identity_history_completion_comparison.csv" \
  --baseline FullMHTIdentityHistory \
  --candidate FullMHTIdentityHistoryCompletion025 \
  --candidate FullMHTIdentityHistoryCompletion050 \
  --candidate FullMHTIdentityHistoryCompletion100 \
  --output "$COMP/full_mht_identity_history_completion_decision.md"

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
| `terminal_completion_stable_gain` | terminal complete-history objective can be considered for the combined row |
| `terminal_completion_single_weight_gain` | terminal objective is exploratory, not promotable |
| `terminal_completion_ties_baseline` | terminal objective supports the story but does not improve the row |
| `incomplete` | rerun the missing manifest, sensitivity, exposure, or no-GT test artifact |

The branch should not claim an original FullMHT method row until this bundle has
been run and recorded.
