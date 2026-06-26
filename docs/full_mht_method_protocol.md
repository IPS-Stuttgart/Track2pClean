# FullMHT Complete-History Method Protocol, 2026-06-26

This document defines the paper-facing bar for promoting FullMHT from an
exploratory Track2p cleanup experiment to an original method row.

The core claim is not that MHT is a different way to veto one bad edge.  The
claim is that longitudinal calcium-imaging tracking can be formulated as Bayesian
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
   switches, gap reactivations, and growth-history prediction are explicit
   history events with costs or likelihoods.
4. **Complete-history objective**: terminal selection or scan-time pruning may
   prefer a lower local scan score when the complete identity history is more
   plausible.

The branch now implements hooks for all four terms.  The remaining question is
not whether the code can express a method; it is whether the frozen benchmark and
exposure artifacts prove that the method does something useful and stable.

## Current Evidence Map

| layer | current status | evidence | decision |
| --- | --- | --- | --- |
| Full scan-assignment beam | implemented | `track2p-policy-full-mht` | keep |
| Greedy-vs-MHT conflict | constructed positive and reference-independent | `track2p-policy-full-mht-conflict-demo` | use as method invariant, not benchmark proof |
| Scan-history conflict | constructed positive | `full_mht_scan_history_conflict_demo` | use as label-free history-search invariant |
| Fixed prior-veto hazard | first positive FullMHT-owned benchmark result | `docs/full_mht_prior_risk_notes.md` | validate, but do not stop here |
| Calibrated association likelihood | implemented and scan-assignment active | `test_calibrated_likelihood_flips_scan_assignment_from_local_overlap` | required method-layer invariant |
| Calibrated prior-edge survival | implemented | `FullMHTPriorSurvival` rows and `full_mht_prior_survival_model.py` | run prior-survival bundle |
| No-prior continuation likelihood | implemented and can open continuation over death | `test_no_prior_continuation_likelihood_opens_scan_assignment_over_death` | run as birth/death probe |
| Growth-history prediction | implemented and scan-time active | `benchmarks/full_mht_growth_history_prediction_probe_manifest.json` | run benchmark plus exposure gate |
| Local-context ablation | frozen | `FullMHTIdentityHistoryNoLocalContext` | required control for candidate row |
| Terminal complete-history objective | implemented | `benchmarks/full_mht_terminal_completion_probe_manifest.json` | run as complete-objective probe |
| Combined identity-history row | frozen, not yet benchmarked | `benchmarks/full_mht_identity_history_candidate_manifest.json` | current paper-facing candidate |
| Combined identity-history sensitivity | frozen, not yet benchmarked | `benchmarks/full_mht_identity_history_sensitivity_manifest.json` | required before promotion |
| Combined scan-history pruning add-on | frozen, not yet benchmarked | `benchmarks/full_mht_identity_history_scan_pruning_manifest.json` | can enter only if full beam beats matching greedy at multiple nearby weights |
| Combined terminal-completion add-on | frozen, not yet benchmarked | `benchmarks/full_mht_identity_history_completion_manifest.json` | can enter only if stable and non-regressing |
| Bundle decision guardrail | implemented | `full_mht_identity_history_bundle_decision.py` | requires `promotable_core_method` with `complete_core_evidence`; `inconsistent_core_evidence` blocks promotion |
| Label-free exposure audit | implemented | `track2p_policy_full_mht_exposure_audit.py` | required before promotion |
| No-GT leakage guard | implemented and widened | `tests/test_full_mht_no_gt_leakage.py` | required before promotion |

## Candidate Method Row

The current paper-facing candidate is:

```text
FullMHTIdentityHistory
```

It combines:

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

It uses the same scan candidates and scoring terms, but sets `beam_width = 1`
and disables identity-diverse beam retention.  Full-beam retention is label-free
and preserves distinct scan-history event signatures: no-prior continuations,
prior-successor switches, missed prior successors, missed tracks, gap
reactivations, and terminal missingness.  The full beam must beat this row on
complete-track F1 micro without any pairwise/complete micro or macro F1 loss
before the benchmark can claim that MHT history search matters on real data.

The canonical manifest also includes:

```text
FullMHTIdentityHistoryNoLocalContext
```

This row is identical to the candidate except `local_deformation_weight = 0.0`.
The candidate must not fall below this control on any reported pairwise or
complete-track micro/macro metric; otherwise the local-neighborhood term has not
earned its place in the method.

A separate probe tests whether scan-time history pruning should be added to the
combined method:

```text
IdentityHistoryScanPruning025
IdentityHistoryScanPruning050
IdentityHistoryScanPruning100
```

Each row has a matching greedy row with the same `scan_motion_history_weight`.
The add-on can enter the paper-facing method only if at least two nearby weights
improve complete-track F1 over their matching greedy rows, no tested neighboring
weight regresses against its greedy row, and no scan-pruning row regresses against
`FullMHTIdentityHistory` on any reported pairwise or complete-track micro/macro
metric.  This is the real-data counterpart of the scan-history conflict witness:
it asks whether MHT preserves better partial identity histories during beam
survival, not only at terminal reranking time.

A separate probe tests whether the terminal complete-history objective should be
added to the combined method:

```text
FullMHTIdentityHistoryCompletion025
FullMHTIdentityHistoryCompletion050
FullMHTIdentityHistoryCompletion100
```

These rows are identical to `FullMHTIdentityHistory` except for
`terminal_incomplete_history_weight`.  They are not a new default.  They become a
candidate only if at least two nearby weights improve complete-track F1 without
pairwise-F1 loss and no tested neighboring weight regresses pairwise or
complete-track F1.

## Promotion Gates

FullMHT can be promoted as a paper method only after these gates pass:

| gate | required evidence |
| --- | --- |
| Manifest comparison | `FullMHTIdentityHistory` beats `FullMHTGreedyIdentityHistory` on complete-track F1 micro with no regression in pairwise or complete-track micro/macro F1 |
| Required controls | `FullMHTIdentityHistory` does not fall below `Track2p`, `FullMHTPrior2`, `FullMHTPriorSurvival`, `FullMHTNoPriorContinuation100`, or `FullMHTIdentityHistoryNoLocalContext` on any reported pairwise or complete-track micro/macro metric |
| Conflict witness | constructed FullMHT-vs-greedy witness passes and selected paths remain unchanged when only the evaluation reference is altered |
| Method-layer invariants | calibrated likelihood changes scan assignment from local-overlap-only behavior; no-prior continuation likelihood can choose continuation over death; growth-history prediction can flip a scan assignment to a coherent history; scan-history pruning rejects a locally attractive but history-incoherent continuation |
| Sensitivity | `benchmarks/full_mht_identity_history_sensitivity_manifest.json` reports `stable_plateau`, with each passing variant non-regressing on all reported micro/macro metrics |
| Scan-history pruning add-on | optional scan-pruning variant reports `scan_pruning_stable_complete_history_gain`, with at least two matching-greedy gains and no regression against greedy or the central identity-history row |
| Exposure | label-free exposure audit reports `bounded_exposure` with active but rare prior-survival, no-prior-continuation, and growth-history signals |
| No-GT leakage | tests confirm method layers do not read `edge_status_against_gt`, `pairwise_delta_if_removed`, `complete_delta_if_removed`, reference identity, or manual-GT status |
| Terminal objective | optional completion variant reports `terminal_completion_stable_gain`, meaning at least two gains and no regressing neighbor in the tested weight neighborhood |
| Bundle decision | final bundle artifact reports `promotable_core_method` with `complete_core_evidence`; `inconsistent_core_evidence` means a stale or contradictory artifact must be rerun |
| Reporting | complete-track and pairwise metrics are reported together, with micro/macro variants |

If the identity-history beam ties its greedy row, regresses on any reported
metric, or improves only pairwise F1, the benchmark does not yet prove a
complete-history search advantage.  In that case FullMHT remains an architecture
and constructed-conflict story, not the headline real-data method row.

## Non-Promotion Conditions

Do not present FullMHT as a final method if any of the following remain true:

- The frozen identity-history manifest has not been run.
- The candidate row ties or loses to its matching greedy ablation.
- The candidate row is worse than a required control on any reported pairwise or
  complete-track micro/macro F1 metric, including the no-local-context control.
- The sensitivity neighborhood is a single-weight spike or hides a macro-metric
  regression.
- The exposure audit shows broad non-prior continuations, broad prior-survival
  penalties, broad growth-history penalties, or many prior switches.
- The final bundle artifact reports `inconsistent_core_evidence`, even if a
  copied or stale top-level status says the row is promotable.
- Scan-history pruning improves only one weight, improves only pairwise F1, loses
  to the matching greedy row, or regresses against the central identity-history
  row.
- The terminal complete-history objective improves only at one fragile weight,
  damages pairwise F1, or damages complete-track F1 at a neighboring tested
  weight.
- Deterministic edge gating over the same candidates produces the same behavior
  without any history-level conflict or history-level benefit.
- The constructed conflict witness changes selected paths when only the reporting
  reference is changed.
- The paper text cannot distinguish the benchmark row from post-hoc growth-veto
  cleanup.

## Validation Recipes

The current validation recipe is concentrated in:

```text
docs/full_mht_identity_history_validation.md
```

Supporting probes and background notes live in:

```text
docs/full_mht_method_invariant_checklist.md
docs/full_mht_prior_survival_validation.md
docs/full_mht_no_prior_continuation_likelihood.md
docs/full_mht_terminal_completion_objective.md
docs/full_mht_growth_history_prediction.md
docs/full_mht_identity_history_scan_pruning.md
docs/full_mht_label_free_exposure_audit.md
docs/full_mht_manifest_integration_notes.md
```

Run the identity-history bundle first.  It is the only bundle that currently asks
the paper-critical question directly: does full MHT identity-history search beat
an equivalent greedy local history under frozen, label-free settings?  Run the
scan-pruning add-on afterward; it can strengthen the method only if the central
identity-history row already passes its own gates.
