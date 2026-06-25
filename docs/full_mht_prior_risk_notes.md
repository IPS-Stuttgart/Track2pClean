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
