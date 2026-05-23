# Track2p-policy min result and next workstream

The Track2p-policy minimum-threshold row is now the primary BayesCaTrack result
for the Track2p-style benchmark.  It keeps the Track2p inductive bias that the
plain BayesCaTrack global-assignment rows were missing: hard Suite2p cell
filtering, consecutive affine registration, Hungarian registered-IoU matching,
minimum-thresholded links, and first-session propagation.

## Current result

On the three-subject benchmark run, Track2p-policy min is effectively at
Track2p-level pairwise performance and is better on complete-track F1:

| method | pairwise F1 micro | complete-track F1 micro |
| --- | ---: | ---: |
| Track2p | 0.965 | 0.924 |
| Track2p-policy min | 0.961 | 0.933 |
| Track2p-policy DP | 0.841 | 0.866 |

This changes the improvement target.  The large original gap was caused mostly
by the global-assignment/cost policy mismatch, not by data loading, scoring, or
basic registration.  Once BayesCaTrack uses the Track2p-style matching policy,
complete-track F1 is already above Track2p.

## Interpretation

Track2p-policy min has higher recall than Track2p, but slightly lower pairwise
precision:

| method | pairwise TP | pairwise FP | pairwise FN | complete TP | complete FP | complete FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Track2p | 581 | 18 | 24 | 55 | 4 | 5 |
| Track2p-policy min | 586 | 28 | 19 | 56 | 4 | 4 |

The next useful target is therefore prune-only: remove a small number of
false-positive policy edges without reducing true positives or the complete-track
advantage.

## DP status

The current DP rescue variant is experimental.  It should not be included in the
default guarded benchmark comparison because the observed DP runs add false
positives and lose policy-supported ground-truth edges.  It remains available as
an opt-in diagnostic via:

```bash
TRACK2P_INCLUDE_POLICY_DP_EXPERIMENT=true \
  poetry run python .github/scripts/run_track2p_benchmark.py
```

or directly:

```bash
bayescatrack benchmark track2p-policy-dp \
  --data <data-root> \
  --reference <manual-gt-root> \
  --reference-kind manual-gt \
  --threshold-method min \
  --iou-distance-threshold 12 \
  --row-top-k 2 \
  --rescue-min-iou 0.10 \
  --threshold-rescue-margin 0.15
```

## Next step

Use the audit command to export a duplicate-aware edge ledger for policy-min:

```bash
bayescatrack benchmark track2p-policy-audit \
  --data <data-root> \
  --reference <manual-gt-root> \
  --reference-kind manual-gt \
  --threshold-method min \
  --iou-distance-threshold 12 \
  --output results/track2p_policy_edge_ledger.csv
```

The prune-only success criterion is: keep complete-track F1 at or above the
policy-min result while reducing pairwise false positives toward the Track2p
count.
