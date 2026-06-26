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

The current branch implements all four hooks, but only the proposal-prior plus
prior-veto hazard has produced a positive benchmark row so far.

## Current Evidence Map

| layer | current status | evidence | decision |
| --- | --- | --- | --- |
| Full scan-assignment beam | implemented | `track2p-policy-full-mht` | keep |
| Greedy-vs-MHT conflict | constructed positive | `track2p-policy-full-mht-conflict-demo` | use as method intuition |
| Calibrated association likelihood | implemented, benchmark-negative | `docs/full_mht_calibrated_likelihood_notes.md` | keep as architecture, not row |
| Identity dynamics penalties | implemented, mostly collapse to proposal solution | `track2p_prior_*` diagnostics | keep |
| Identity-diverse beam | implemented, exposes cleaner alternatives | calibrated-likelihood notes | keep |
| Prior-edge survival hazard | first positive FullMHT-owned result | `docs/full_mht_prior_risk_notes.md` | freeze and validate |
| Manifest-level reproduction | manifest + adapter committed | `benchmarks/full_mht_prior_veto_manifest.json` | run on server |

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

## Non-Promotion Conditions

Do not present FullMHT as a final method if any of the following remain true:

- The frozen manifest cannot reproduce the positive row.
- The positive row depends on inspecting manual-GT audit columns.
- Exposure audit shows the prior-veto hazard fires broadly across non-GT
  Track2p-style subjects.
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
| Manifest reproduction | `bayescatrack benchmark suite benchmarks/full_mht_prior_veto_manifest.json` reproduces Track2p, FullMHTPrior2, and FullMHTPriorVetoScaled rows |
| No-GT leakage | tests confirm scoring functions do not read `edge_status_against_gt`, `pairwise_delta_if_removed`, `complete_delta_if_removed`, reference identity, or manual-GT status |
| Exposure audit | all Track2p-style subjects report rare prior-veto hazards and no subject receives a broad set of missed prior successors |
| Sensitivity | immediate threshold neighbors keep complete-track F1 >= FullMHTPrior2, no selected true-positive removals, and selected edits remain tiny |
| Greedy ablation | deterministic local selection over the same scan candidates is compared against FullMHT history selection |
| Conflict demonstration | at least the constructed conflict demo, and ideally one real benchmark subject, shows a locally better edge loses to a better complete history |
| Reporting | complete-track and pairwise metrics are reported together, with micro/macro variants where relevant |

## Next Implementation Jump

The prior-veto row is promising but still too close to a gated hazard. The next
real method jump should replace the fixed hazard pocket with a calibrated
survival probability for Track2p prior edges:

```text
log p(edge survives | label-free diagnostics)
```

Candidate features should include:

- registered IoU and shifted IoU
- growth residual and growth Mahalanobis
- endpoint cell probabilities and minimum endpoint confidence
- area/shape ratio
- row/column assignment ranks
- terminal-edge and complete-component indicators
- local-neighbor deformation consistency

The MHT score should then combine:

```text
proposal prior + association likelihood + prior-edge survival likelihood
+ missed-detection / death likelihood + terminal identity-history objective
```

This would reduce the current hand-gated prior-veto pocket to a calibrated model
layer, making MHT responsible for full identity-history selection rather than for
validating a single cleanup edit.

## Server Commands To Run Next

Use the Python 3.12 environment on the benchmark server:

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
  tests/test_track2p_policy_full_mht_conflict_demo.py \
  tests/test_track2p_policy_full_mht_growth_prior.py::test_full_mht_prior_veto_scoring_does_not_read_gt_audit_columns

OUT="$REPO/results/full_mht_prior_veto_manifest_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
"$PY" -m bayescatrack benchmark suite \
  benchmarks/full_mht_prior_veto_manifest.json \
  --output-dir "$OUT" \
  --summary-format table
```

Record the output directory and comparison table in
`docs/full_mht_manifest_integration_notes.md` after the run.
