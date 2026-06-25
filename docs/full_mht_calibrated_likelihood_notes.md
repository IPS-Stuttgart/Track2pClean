# Full MHT Calibrated Likelihood Notes, 2026-06-25

The full scan-assignment MHT runner now has an opt-in calibrated association
score mode:

```text
--association-score-mode calibrated-likelihood
--association-likelihood-weight 1.0
--association-likelihood-clip 4.0
```

The mode is label-free. It uses mutual high-overlap, growth-consistent edges as
pseudo-positive anchors and the local scan candidate background as the negative
reference distribution. The resulting robust Gaussian log-likelihood-ratio score
is available as a scan-assignment score matrix; the default `heuristic` scoring
path is unchanged.

## Validation

Fresh server clone at commit `b6f23b857e51e30da5dcacb934fe36ced092e80f`:

```text
35 passed in 0.52s
```

The new CLI flags are registered:

```text
--association-score-mode {heuristic,calibrated-likelihood}
--association-likelihood-weight ASSOCIATION_LIKELIHOOD_WEIGHT
--association-likelihood-clip ASSOCIATION_LIKELIHOOD_CLIP
```

## Probe Results

Unbounded initial probe:

`/home/florianpfaff/codex-runs/BayesCaTrack/results/full_mht_calibrated_likelihood_probe_20260625_234759`

| row | pairwise F1 micro | complete-track F1 micro |
| --- | ---: | ---: |
| Track2p | 0.965116 | 0.924370 |
| FullMHTPrior2 | 0.965116 | 0.924370 |
| CalibratedPrior2 | 0.895618 | 0.796748 |

Bounded likelihood probe, with `--association-likelihood-clip 4.0`:

`/home/florianpfaff/codex-runs/BayesCaTrack/results/full_mht_calibrated_likelihood_clip_probe_20260625_235933`

| row | pairwise F1 micro | complete-track F1 micro |
| --- | ---: | ---: |
| Track2p | 0.965116 | 0.924370 |
| FullMHTPrior2 | 0.965116 | 0.924370 |
| CalibratedClip4Prior2 | 0.935663 | 0.896000 |

## Diagnostic Readout

For the bounded calibrated row, the best hypotheses selected 42 non-prior edges:

| row | selected prior edges | selected non-prior edges | missed prior successors |
| --- | ---: | ---: | ---: |
| FullMHTPrior2 | 599 | 0 | 0 |
| CalibratedClip4Prior2 | 618 | 42 | 0 |

By subject for `CalibratedClip4Prior2`:

| subject | selected prior edges | selected non-prior edges | missed prior successors |
| --- | ---: | ---: | ---: |
| jm038 | 203 | 34 | 0 |
| jm039 | 204 | 7 | 0 |
| jm046 | 211 | 1 | 0 |

The calibrated likelihood did not mainly fail by dropping proposal successors. It
failed by adding extra non-prior continuations, especially in `jm038`.

## Decision

Do not promote the calibrated-likelihood FullMHT row yet. The implementation is a
useful method layer because it replaces hand-weighted local scoring with a
label-free likelihood-ratio score, but the pseudo-anchor calibration is still too
permissive without a stronger birth/death or non-prior continuation model.

Next method step: keep the calibrated association score, but add an explicit
non-prior continuation prior or birth/death model so MHT can open new histories
only when the full identity hypothesis justifies them.
