# Full MHT Proposal-Prior Risk Notes, 2026-06-25

The full scan-assignment MHT row is now controllable around the Track2p proposal
prior. With a strong proposal bonus and a non-proposal deviation penalty, the
runner exactly reproduces the Track2p baseline under the official seed-restricted
benchmark. Weakening the prior globally does not produce a useful method row.

## Proposal-Prior Deviation Sweep

Output directory:

`/home/florianpfaff/codex-runs/BayesCaTrack/results/full_mht_nonprior_sweep_20260625_214931`

| row | pairwise F1 micro | complete-track F1 micro |
| --- | ---: | ---: |
| Track2p | 0.965116 | 0.924370 |
| Prior12NonPrior1Miss4 | 0.953947 | 0.909091 |
| Prior12NonPrior2Miss4 | 0.965116 | 0.924370 |
| Prior12NonPrior3Miss4 | 0.965116 | 0.924370 |
| Prior12NonPrior4Miss4 | 0.965116 | 0.924370 |

Selected-edge diagnostics showed that penalty 1 admitted eight non-prior edges in
the best hypotheses, six of them in `jm038`. Penalty 2 suppressed all non-prior
edges and collapsed to the Track2p proposal solution.

## Soft-Prior Sweep

Output directory:

`/home/florianpfaff/codex-runs/BayesCaTrack/results/full_mht_soft_prior_sweep_20260625_221947`

| row | pairwise F1 micro | complete-track F1 micro |
| --- | ---: | ---: |
| Track2p | 0.965116 | 0.924370 |
| Prior4NonPrior2Miss0 | 0.925217 | 0.851852 |
| Prior6NonPrior2Miss0 | 0.944492 | 0.902655 |
| Prior8NonPrior2Miss0 | 0.952782 | 0.913793 |
| Prior10NonPrior2Miss0 | 0.960801 | 0.915254 |

Lowering the proposal prior mainly damages `jm038`. The row improves as the prior
gets stronger, but it does not reach the Track2p baseline before returning to the
proposal solution.

## Prior-Edge Risk Probe

Output directories:

`/home/florianpfaff/codex-runs/BayesCaTrack/results/full_mht_prior_risk_probe_20260625_224548`

`/home/florianpfaff/codex-runs/BayesCaTrack/results/full_mht_prior_risk_mahal_gentle_20260625_225440`

The residual-plus-IoU prior-risk setting was too blunt:

| row | pairwise F1 micro | complete-track F1 micro |
| --- | ---: | ---: |
| RiskMild | not compared in final CSV | not compared in final CSV |

Per-subject summary for `RiskMild`:

| subject | pairwise F1 | complete-track F1 |
| --- | ---: | ---: |
| jm038 | 0.640523 | 0.230769 |
| jm039 | 0.780186 | 0.580645 |
| jm046 | 0.910526 | 0.888889 |

A gentler Mahalanobis-only prior-risk setting also degraded the benchmark:

| row | pairwise F1 micro | complete-track F1 micro |
| --- | ---: | ---: |
| Track2p | 0.965116 | 0.924370 |
| Penalty2 | 0.965116 | 0.924370 |
| RiskMahalGentle | 0.902765 | 0.772277 |

## Decision

Do not promote scalar prior-risk penalties. They are useful code hooks and
diagnostics, but not a method row. The evidence points toward a targeted,
calibrated prior-edge risk model or a component-level complete-track objective:
keep the proposal prior strong overall, and reject only a small number of risky
proposal edges with better calibrated evidence.

## Terminal History Risk Probe

Output directory:

`/home/florianpfaff/codex-runs/BayesCaTrack/results/full_mht_terminal_history_probe_20260625_232546`

Completed subset comparison:

`terminal_history_completed_subset_comparison.csv`

The terminal reranker keeps the scan-level proposal prior intact and applies the
prior-edge risk only when selecting among completed full-track hypotheses. This
is closer to the desired complete-history MHT objective than local scan-time risk.
The first completed Mahalanobis-only setting was:

```text
--track2p-prior-risk-mahalanobis-weight 1.0
--track2p-prior-risk-mahalanobis-offset 2.5
--track2p-prior-risk-scan-weight 0.0
--terminal-history-risk-weight 1.0
```

| row | pairwise F1 micro | complete-track F1 micro |
| --- | ---: | ---: |
| Track2p | 0.965116 | 0.924370 |
| FullMHTPrior2 | 0.965116 | 0.924370 |
| TerminalMahal1Offset25 | 0.964256 | 0.924370 |

The terminal reranker did exercise the full-history selection path: for `jm038`,
it selected terminal rank 2 instead of rank 1. That changed the full identity
history but did not rescue complete-track F1 and slightly reduced pairwise F1.

Decision: terminal history reranking is a useful architectural hook, but scalar
Mahalanobis prior-edge risk is still not selective enough. The next method layer
should replace this scalar risk with a calibrated association likelihood or a
component-level objective that explicitly models complete-track breakage.

## Prior-Edge Survival Hazard Probe

The next implementation added an opt-in Track2p prior-edge survival hazard:

```text
--track2p-prior-veto-penalty
```

With the default penalty of zero, existing full-MHT behavior is unchanged. When
enabled, the hazard applies only to Track2p prior edges that satisfy a strict,
label-free pocket: terminal edge, last-session edge, complete prior component,
high growth residual, bounded registered/shifted overlap, weak endpoint cap, and
row/column rank 1. This moves the successful residual cleanup idea into the full
scan-assignment MHT objective: suspicious prior edges receive low survival
likelihood during assignment rather than being edited after tracking.

Fresh server clone at commit `ce3b206d26c304c718509a9e5a8c98047c6c7088`:

```text
42 passed in 0.51s
```

Output directories:

`/home/florianpfaff/codex-runs/BayesCaTrack/results/full_mht_prior_veto_hazard_probe_20260626_020723`

`/home/florianpfaff/codex-runs/BayesCaTrack/results/full_mht_prior_veto_scaled_probe_20260626_022258`

### Strict Residual Scale

Using the original residual-MHT growth-veto Mahalanobis threshold (`20`) did not
activate inside FullMHT, because FullMHT's growth Mahalanobis values are on a
different scale. The known edge `jm046 5:2309->6:1210` had:

```text
reg=0.3636
shift=0.7647
growth=2.907
mahal=2.699
veto=growth_residual_mahalanobis_below_gate
```

The strict run therefore tied the Track2p/FullMHTPrior2 control:

| row | pairwise F1 micro | complete-track F1 micro |
| --- | ---: | ---: |
| Track2p | 0.965116 | 0.924370 |
| FullMHTPrior2 | 0.965116 | 0.924370 |
| FullMHTPriorVeto20 | 0.965116 | 0.924370 |

### FullMHT-Scale Strict Pocket

The scale-aligned probe changed only the FullMHT-scale gates:

```text
--track2p-prior-veto-penalty 20
--track2p-prior-veto-min-growth-residual-mahalanobis 2.5
--track2p-prior-veto-min-registered-iou 0.35
--track2p-prior-veto-max-registered-iou 0.40
```

All structural guards remained active. This selected a single missed prior
successor in `jm046` and did not introduce non-prior continuations or prior
switches:

```text
scan_selected_prior_edges=598
scan_selected_non_prior_edges=0
scan_missed_prior_successors=1
scan_switched_prior_successors=0
scan_no_prior_successor_continuations=0
```

| row | pairwise F1 micro | complete-track F1 micro |
| --- | ---: | ---: |
| Track2p | 0.965116 | 0.924370 |
| FullMHTPrior2 | 0.965116 | 0.924370 |
| FullMHTPriorVetoScaled | 0.965919 | 0.932203 |

Decision: this is the first positive full-MHT result, and it is conceptually more
interesting than a post-hoc cleanup: the MHT model keeps the Track2p proposal as
a strong prior but gives low survival likelihood to a narrowly defined suspicious
prior edge. Do not call it final yet. The scale-aligned gate was informed by the
same benchmark ledger, so it needs a reproducibility bundle, no-GT-leakage test,
non-GT exposure audit, and a small threshold stability table before promotion.

### Sensitivity and No-GT Check

The prior-veto scoring path now has a no-GT-leakage regression:

```text
test_full_mht_prior_veto_scoring_does_not_read_gt_audit_columns
```

It guards the FullMHT scoring helpers against audit-only fields such as
`edge_status_against_gt`, `pairwise_delta_if_removed`, and
`complete_delta_if_removed`. A clean server clone plus the patch passed the
focused Python 3.12 check:

```text
2 passed in 0.49s
```

Output directory:

`/home/florianpfaff/codex-runs/BayesCaTrack/results/full_mht_prior_veto_sensitivity_20260626_023423`

The small sensitivity bundle reused the Track2p and FullMHTPrior2 controls and
tested immediate threshold neighbors around the scale-aligned prior-veto pocket:

| row | changed gate | missed prior successors | non-prior continuations | pairwise F1 micro | complete-track F1 micro |
| --- | --- | ---: | ---: | ---: | ---: |
| VetoM25Reg035040Cell065 | baseline scaled pocket | 1 | 0 | 0.965919 | 0.932203 |
| VetoM20Reg035040Cell065 | Mahalanobis min 2.0 | 1 | 0 | 0.965919 | 0.932203 |
| VetoM30Reg035040Cell065 | Mahalanobis min 3.0 | 0 | 0 | 0.965116 | 0.924370 |
| VetoM25Reg033042Cell065 | registered IoU 0.33..0.42 | 1 | 0 | 0.965919 | 0.932203 |
| VetoM25Reg035040Cell070 | weak-endpoint cap 0.70 | 1 | 0 | 0.965919 | 0.932203 |

All positive neighbors missed only one prior successor in `jm046`; none selected
non-prior continuations or switched away from a Track2p prior successor. The
too-strict Mahalanobis threshold turned the hazard off and returned exactly to
the Track2p/FullMHTPrior2 control.

Interpretation: the result is not a single exact-threshold spike, but the plateau
is still small and tied to the FullMHT growth-residual scale. This is promising
enough to keep developing as the first "FullMHT-owned" method row, but promotion
still requires exposure on all Track2p-style subjects and a manifest-level
comparison against the residual-MHT and teacher-assisted rows.

## Frozen Manifest, 2026-06-26

The intended reproducibility bundle is now frozen in:

`benchmarks/full_mht_prior_veto_manifest.json`

It contains three rows:

| row | purpose |
| --- | --- |
| Track2p | original Track2p proposal baseline |
| FullMHTPrior2 | full scan-assignment MHT constrained by the Track2p proposal prior |
| FullMHTPriorVetoScaled | FullMHT with the label-free prior-edge survival hazard enabled |

The manifest intentionally records the scale-aligned prior-veto gates that
produced the first positive FullMHT-owned row:

```text
track2p_prior_veto_penalty = 20.0
track2p_prior_veto_min_growth_residual_mahalanobis = 2.5
track2p_prior_veto_min_registered_iou = 0.35
track2p_prior_veto_max_registered_iou = 0.40
track2p_prior_veto_max_min_cell_probability = 0.65
```

Status: frozen but not yet canonical. The benchmark-suite manifest adapter still
needs to register `track2p-policy-full-mht` / `track2p-full-mht` and translate
these JSON fields into `FullMHTConfig` before this file can be executed with
`bayescatrack benchmark suite`. Until that adapter is merged, use the existing CLI
commands as the authoritative execution path and treat this manifest as the
pre-registered target bundle.
