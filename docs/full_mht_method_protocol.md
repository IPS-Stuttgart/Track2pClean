# FullMHT Complete-History Method Protocol, 2026-06-26

This document defines the paper-facing bar for promoting FullMHT from an
exploratory Track2p cleanup experiment to an original method row.

The core claim is not that MHT is a prettier way to veto one bad edge. The claim
is that longitudinal calcium-imaging tracking can be formulated as Bayesian
identity-history selection: pairwise-good local links can still produce
complete-track failures, and a bounded MHT beam can compare whole identity
histories rather than isolated links.

## Method Hypothesis

A defensible FullMHT method should combine four label-free terms:

1. **Proposal prior**: Track2p edges are treated as strong survival proposals,
   not as immutable ground truth.
2. **Association likelihood**: registered IoU, shifted IoU, area/shape, cell
   probability, centroid distance, growth residual, growth Mahalanobis, and local
   deformation enter as likelihood-ratio evidence rather than hand-picked GT
   ledgers.
3. **Identity dynamics**: missed detections, no-prior continuations, prior
   switches, and gap reactivations are explicit history events with costs or
   likelihoods.
4. **Complete-history objective**: terminal selection may prefer a lower local
   scan score when the complete identity history is more plausible.

The current branch implements all four hooks. The first positive benchmark row is
still the fixed prior-veto hazard, but the branch now also has a calibrated
prior-edge survival likelihood row ready for manifest-level evaluation.

## Current Evidence Map

| layer | current status | evidence | decision |
| --- | --- | --- | --- |
| Full scan-assignment beam | implemented | `track2p-policy-full-mht` | keep |
| Greedy-vs-MHT conflict | constructed positive | `track2p-policy-full-mht-conflict-demo` | use as method intuition |
| Calibrated association likelihood | implemented, benchmark-negative | `docs/full_mht_calibrated_likelihood_notes.md` | keep as architecture, not row |
| Identity dynamics penalties | implemented, mostly collapse to proposal solution | `track2p_prior_*` diagnostics | keep |
| Identity-diverse beam | implemented, exposes cleaner alternatives | calibrated-likelihood notes | keep |
| Fixed prior-veto hazard | first positive FullMHT-owned result | `docs/full_mht_prior_risk_notes.md` | freeze and validate |
| Calibrated prior-edge survival | integrated, not yet benchmarked | `full_mht_prior_survival_model.py`, `FullMHTPriorSurvival` manifest row | run on server |
| Manifest-level reproduction | manifest + adapter committed | `benchmarks/full_mht_prior_veto_manifest.json` | run on server |
| Sensitivity/exposure bundle | committed, not yet run | `benchmarks/full_mht_prior_survival_sensitivity_manifest.json`, `docs/full_mht_prior_survival_validation.md` | run on server |

## Conflict Demonstrations

`track2p-policy-full-mht-conflict-demo` now provides two controlled ablations:

- `local-edge-dead-end`: a locally stronger first edge leads to a missed final
  observation, while the MHT beam preserves the weaker first edge until the later
  continuation makes the complete history win.
- `pairwise-good-complete-bad`: the same principle is embedded among many stable
  tracks, so the greedy result remains pairwise-good but creates a wrong complete
  identity. FullMHT keeps the alternative middle edge alive and recovers the
  complete path.

The second scenario is the method-story invariant: pairwise-good local assignment
can still be complete-track-bad, and a history beam can fix the identity history
without reading GT labels. It is not benchmark evidence; it is the executable
ablation that shows what the full MHT architecture can do that deterministic
local selection cannot.

Run it with:

```bash
python -m bayescatrack.experiments.track2p_policy_full_mht_conflict_demo \
  --scenario pairwise-good-complete-bad
```

## Current Positive Row

The current positive non-teacher FullMHT row is:

```text
FullMHTPriorVetoScaled
```

It keeps the Track2p proposal prior strong, but gives low survival likelihood to
a narrowly defined suspicious Track2p prior edge. In the canonical lightweight
manual-GT benchmark it improved the FullMHT/Track2p proposal control:

| row | pairwise F1 micro | complete-track F1 micro |
| --- | ---: | ---: |
| Track2p | 0.965116 | 0.924370 |
| FullMHTPrior2 | 0.965116 | 0.924370 |
| FullMHTPriorVetoScaled | 0.965919 | 0.932203 |

The important methodological distinction from residual cleanup is that the bad
prior edge is penalized during scan-assignment history selection. It is not
removed after a completed Track2p solution is scored.

## Candidate Method Row

The next paper-facing candidate is:

```text
FullMHTPriorSurvival
```

It uses the same Track2p proposal prior and scan-assignment beam, but replaces
the hand-gated prior-veto pocket with a calibrated, label-free prior-edge survival
log-likelihood ratio. The manifest row is now frozen enough to run, but no
benchmark result has been recorded yet.

## Non-Promotion Conditions

Do not present FullMHT as a final method if any of the following remain true:

- The frozen manifest cannot reproduce the positive row.
- The candidate survival row does not match or improve the fixed prior-veto row.
- The positive row depends on inspecting manual-GT audit columns.
- Exposure audit shows the prior-veto or survival hazard fires broadly across
  non-GT Track2p-style subjects.
- A nearby threshold perturbation selects true-positive removals or causes
  complete-track loss.
- Deterministic edge gating over the same candidates produces exactly the same
  behavior without any history-level conflict or history-level benefit.
- The paper text cannot distinguish the benchmark row from post-hoc growth-veto
  cleanup.

## Promotion Gates

FullMHT can be promoted as a paper method only after these gates pass:

| gate | required evidence |
| --- | --- |
| Manifest reproduction | `bayescatrack benchmark suite benchmarks/full_mht_prior_veto_manifest.json` reproduces Track2p, FullMHTPrior2, FullMHTPriorVetoScaled, and FullMHTPriorSurvival rows |
| No-GT leakage | tests confirm scoring functions do not read `edge_status_against_gt`, `pairwise_delta_if_removed`, `complete_delta_if_removed`, reference identity, or manual-GT status |
| Exposure audit | all Track2p-style subjects report rare prior-veto/survival hazards and no subject receives a broad set of missed prior successors |
| Sensitivity | `benchmarks/full_mht_prior_survival_sensitivity_manifest.json` shows nearby survival weights/clips/pseudo-label settings do not collapse pairwise or complete-track metrics |
| Greedy ablation | deterministic local selection over the same scan candidates is compared against FullMHT history selection |
| Conflict demonstration | at least the constructed conflict demo, and ideally one real benchmark subject, shows a locally better edge loses to a better complete history |
| Reporting | complete-track and pairwise metrics are reported together, with micro/macro variants where relevant |

## Implemented Method Jump

The fixed prior-veto row is promising but still too close to a gated hazard. The
branch now implements the next method jump: a calibrated survival probability for
Track2p prior edges:

```text
log p(edge survives | label-free diagnostics)
```

Candidate features include:

- registered IoU and shifted IoU
- growth residual and growth Mahalanobis
- endpoint cell probabilities and minimum endpoint confidence
- area/shape ratio
- row/column assignment ranks
- terminal-edge and complete-component indicators
- local-neighbor deformation consistency

The MHT score can now combine:

```text
proposal prior + association likelihood + prior-edge survival likelihood
+ missed-detection / death likelihood + terminal identity-history objective
```

This reduces the current hand-gated prior-veto pocket to a calibrated model layer
when `FullMHTPriorSurvival` is enabled, making MHT responsible for full
identity-history selection rather than for validating a single cleanup edit.

## Server Commands To Run Next

Use the Python 3.12 environment on the benchmark server. The complete validation
recipe lives in:

```text
docs/full_mht_prior_survival_validation.md
```

Minimum command bundle:

```bash
REPO=/home/florianpfaff/codex-runs/BayesCaTrack
PY="$REPO/.venv312/bin/python"
cd "$REPO"
git fetch origin
git checkout codex/full-mht-prototype
git reset --hard origin/codex/full-mht-prototype
export PYTHONPATH="$REPO/src"

"$PY" -m pytest -q \
  tests/test_benchmark_manifest_full_mht_integration.py \
  tests/test_full_mht_prior_survival_model.py \
  tests/test_full_mht_prior_survival_integration.py \
  tests/test_track2p_policy_full_mht_conflict_demo.py \
  tests/test_track2p_policy_full_mht_growth_prior.py::test_full_mht_prior_veto_scoring_does_not_read_gt_audit_columns

OUT="$REPO/results/full_mht_prior_survival_manifest_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
"$PY" -m bayescatrack benchmark suite \
  benchmarks/full_mht_prior_veto_manifest.json \
  --output-dir "$OUT" \
  --summary-format table

SENS="$REPO/results/full_mht_prior_survival_sensitivity_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SENS"
"$PY" -m bayescatrack benchmark suite \
  benchmarks/full_mht_prior_survival_sensitivity_manifest.json \
  --output-dir "$SENS" \
  --summary-format table
```

Record the output directories, comparison tables, and promote/keep-exploratory
judgment in `docs/full_mht_prior_survival_validation.md` and
`docs/full_mht_manifest_integration_notes.md` after the run.
