# Track2p-policy prune-only experiment

The current highest-leverage BayesCaTrack direction is the promoted
`track2p-policy` row.  The next safe improvement target is not a rescue method:
it is a conservative post-filter that removes only high-risk policy edges.

`track2p-policy-pruned` keeps the same Track2p-style pipeline as
`track2p-policy` and applies one additional gate after the normal Hungarian IoU
assignment and Track2p thresholding.  A link is removed only when all of these
signals are weak:

1. the assigned IoU is barely above the Track2p-policy threshold,
2. both the row and column have close competing alternatives,
3. area ratio or registered centroid evidence is poor.

The method never adds links.  This is deliberate: complete-track F1 is fragile,
and previous rescue-style experiments lost the policy row's complete-track
advantage.

## Run the opt-in benchmark row

```bash
bayescatrack benchmark track2p-policy-pruned \
  --data ../benchmark-raw-suite2p-subjects \
  --reference ../benchmark-raw-suite2p-subjects \
  --reference-kind manual-gt \
  --threshold-method min \
  --iou-distance-threshold 12 \
  --prune-threshold-margin 0.02 \
  --prune-competition-margin 0.02 \
  --prune-min-area-ratio 0.45 \
  --prune-centroid-distance 10 \
  --output results/track2p_policy_pruned.csv \
  --diagnostics-output results/track2p_policy_pruned_edges.csv \
  --format csv
```

The diagnostics CSV records every threshold-accepted policy edge with:

- assigned IoU,
- threshold value and threshold margin,
- row and column competition margins,
- registered centroid distance,
- area ratio,
- prune decision and prune reason.

## CI usage

Keep the row out of the default guarded comparison until it is validated on the
manual-ground-truth benchmark:

```bash
TRACK2P_INCLUDE_POLICY_PRUNED_EXPERIMENT=true \
poetry run python .github/scripts/run_track2p_benchmark.py
```

## Selection rule

Tune the prune thresholds fold-cleanly.  The selection objective should be:

1. preserve or improve complete-track F1 versus `track2p-policy`,
2. improve pairwise F1 by reducing false-positive policy edges,
3. reject any setting that removes too many true-positive edges.

This keeps the experiment aligned with the narrow remaining gap: retain the
policy row's complete-track advantage while closing the pairwise-F1 gap to
Track2p.
