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
beat this greedy row on complete-track F1 micro without loss on any reported
pairwise or complete-track micro/macro metric.

The canonical manifest also includes the local-context ablation:

```text
FullMHTIdentityHistoryNoLocalContext
```

This row is identical to `FullMHTIdentityHistory` except that
`local_deformation_weight = 0.0`, which makes the calibrated association
likelihood ignore local-neighborhood deformation.  Promotion requires the full
candidate to beat or match this control on every reported pairwise and
complete-track micro/macro metric, so the local-context layer has to earn its
place inside the method rather than merely being present in the implementation.

## Constructed Conflict Witness

The branch also keeps a small label-free conflict witness in
`track2p-policy-full-mht-conflict-demo`.  It is not a benchmark result; it is a
mechanical proof-of-behavior for the method claim.  The demo constructs scan
assignment histories where the locally best edge leads to a later dead end, while
a slightly weaker edge preserves a complete identity once later evidence arrives.

The focused regression requires three invariants:

```text
full MHT beam > greedy beam-width-1 on the dead-end conflict
full MHT beam improves complete-track F1 while greedy remains pairwise-good
selected greedy and full-MHT paths are unchanged when the evaluation reference is altered
```

The last invariant matters because the witness uses a reference matrix only to
report pairwise and complete-track metrics.  It must not influence which identity
history either arm selects.  This witness does not promote the real-data row by
itself; it only protects the claim that the MHT beam can be load-bearing when
identity histories conflict.

## Local-Context Probe

The branch now includes a frozen calibrated local-neighborhood deformation probe:

```text
FullMHTLocalContext000
FullMHTLocalContext025
FullMHTLocalContext050
FullMHTLocalContext100
```

These rows keep the same FullMHT prior setup and calibrated association
likelihood, sweeping only `local_deformation_weight`.  The manifest installs a
local-context likelihood gate so `FullMHTLocalContext000` is a true calibrated
no-local-context ablation.  The probe asks whether a label-free neighborhood
coherence term helps independently of the later identity-history bundle.

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
without pairwise-F1 loss and no tested neighboring weight regresses any reported
pairwise or complete-track micro/macro metric.  A single winning weight, or a win
beside a regressing weight, is treated as exploratory.

## Frozen Artifacts

| artifact | purpose |
| --- | --- |
| `benchmarks/full_mht_identity_history_candidate_manifest.json` | canonical comparison against Track2p, prior-only FullMHT, prior-survival, no-prior continuation, no-local-context identity history, and greedy identity history |
| `benchmarks/full_mht_identity_history_sensitivity_manifest.json` | immediate-neighborhood sensitivity around survival weight, no-prior continuation weight, and growth-history weight |
| `benchmarks/full_mht_identity_history_completion_manifest.json` | complete-history terminal objective probe on top of the combined identity-history row |
| `benchmarks/full_mht_local_context_probe_manifest.json` | calibrated local-neighborhood deformation probe against a no-local-context FullMHT prior baseline |
| `docs/full_mht_method_invariant_checklist.md` | paper-facing checklist tying method claims to required label-free regressions |
| `test_full_mht_method_protocol.py` | regression that keeps the method protocol and invariant checklist from drifting |
| `full_mht_local_context_integration.py` | gates the calibrated local-context likelihood feature when `local_deformation_weight <= 0` |
| `track2p_policy_full_mht_conflict_demo.py` | constructed witness that full-history beam search can beat greedy local assignment in an identity-history conflict |
| `test_track2p_policy_full_mht_conflict_demo.py` | regression for the constructed MHT-vs-greedy conflict witness, including reference-independent path selection |
| `full_mht_local_context_decision.py` | interprets the local-neighborhood deformation probe |
| `full_mht_identity_history_decision.py` | interprets the canonical comparison table, including greedy and no-local-context controls |
| `full_mht_identity_history_promotion_gate.py` | combines canonical decision, sensitivity, and label-free exposure audit |
| `full_mht_terminal_completion_decision.py` | interprets the terminal-completion probe, with row-name overrides for identity-history rows |
| `track2p_policy_full_mht_exposure_audit.py` | runs all Track2p-style subjects without loading references or audit labels |

## Decision Rule

Promote `FullMHTIdentityHistory` only if all of these are true:

- `FullMHTIdentityHistory` has complete-track F1 micro advantage over `FullMHTGreedyIdentityHistory` with no pairwise/complete micro or macro F1 loss.
- It does not fall below `FullMHTIdentityHistoryNoLocalContext` on any reported pairwise or complete-track micro/macro metric.
- It does not fall below `Track2p`, `FullMHTPrior2`, `FullMHTPriorSurvival`, or `FullMHTNoPriorContinuation100` on any reported pairwise or complete-track micro/macro metric.
- The constructed conflict witness regression passes.
- The method-invariant checklist regression passes.
- The sensitivity manifest reports `stable_plateau`, with passing variants non-regressing on all reported metrics.
- The exposure audit reports `bounded_exposure`.
- Prior-survival, no-prior continuation, and growth-history signals are active but not broad.
- The no-GT leakage regression passes.

Promote a terminal-completion variant only if the identity-history row itself
passes those gates and the completion probe reports `terminal_completion_stable_gain`,
which requires a non-regressing immediate weight neighborhood.  If any gate fails,
keep the row exploratory.  A tie against greedy means the benchmark still does
not prove that MHT history search, rather than local scoring, is responsible for
the result.  A loss against the no-local-context control means the calibrated
local-neighborhood layer should be removed or kept exploratory.

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
  tests/test_full_mht_method_protocol.py \
  tests/test_full_mht_prior_survival_model.py \
  tests/test_full_mht_prior_survival_integration.py \
  tests/test_full_mht_no_prior_continuation_model.py \
  tests/test_full_mht_no_prior_continuation_integration.py \
  tests/test_full_mht_growth_history_prediction_integration.py \
  tests/test_full_mht_terminal_completion_decision.py \
  tests/test_full_mht_local_context_manifest.py \
  tests/test_full_mht_local_context_decision.py \
  tests/test_full_mht_local_context_integration.py \
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

LOCAL="$REPO/results/full_mht_local_context_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOCAL"
"$PY" -m bayescatrack benchmark suite \
  benchmarks/full_mht_local_context_probe_manifest.json \
  --output-dir "$LOCAL" \
  --summary-format table

"$PY" -m bayescatrack.experiments.full_mht_local_context_decision \
  "$LOCAL/full_mht_local_context/full_mht_local_context_comparison.csv" \
  --output "$LOCAL/full_mht_local_context_decision.md"

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
| `promotable_after_review` | strong candidate for the paper method row, after recording exact directories and all four metric tables |
| `not_promotable_manifest` | no real-data proof that MHT history search beats greedy local selection, or the candidate regresses on a required micro/macro control |
| `not_promotable_sensitivity` | likely knife-edge, single-setting result, or hidden macro regression |
| `not_promotable_broad_exposure` | model layer fires too broadly on label-free subjects |
| `history_dynamics_stable_gain` | local context or another dynamics probe shows stable complete-track gain without pairwise loss |
| `history_dynamics_single_weight_gain` | layer probe is exploratory, not promotable |
| `terminal_completion_stable_gain` | terminal complete-history objective can be considered only when the tested weight neighborhood has at least two gains and no pairwise or complete-track regression |
| `terminal_completion_single_weight_gain` | terminal objective is exploratory, not promotable |
| `terminal_completion_ties_baseline` | terminal objective supports the story but does not improve the row |
| `incomplete` | rerun the missing manifest, sensitivity, exposure, or no-GT test artifact |

The branch should not claim an original FullMHT method row until this bundle has
been run and recorded.
