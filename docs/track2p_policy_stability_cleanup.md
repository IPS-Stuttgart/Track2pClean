# Track2p-policy stability cleanup

`track2p-policy-stability-cleanup` is a prune-only follow-up to the promoted Track2p-policy row. It starts from the base Track2p-policy prediction and removes only bridges that are not stable under a small ensemble of nearby IoU-distance thresholds. It does not add rescue edges.

The intended use case is the current Track2p-policy error mode: high complete-track recall with a small number of false-positive continuations. A bridge is kept when it appears in enough policy predictions across the configured threshold ensemble; otherwise, the base track is split at that bridge when both resulting fragments remain long enough.

## Example

```bash
poetry run python -m bayescatrack.experiments.track2p_policy_stability_cleanup \
  --data ../benchmark-raw-suite2p-subjects \
  --reference ../benchmark-raw-suite2p-subjects \
  --reference-kind manual-gt \
  --threshold-method min \
  --base-iou-distance-threshold 12 \
  --stability-iou-distance-thresholds 10,12,14 \
  --min-support-fraction 0.667 \
  --min-side-observations 2 \
  --format csv \
  --output results/track2p_policy_stability_cleanup.csv
```

## Interpretation

The row is deliberately conservative:

- The base prediction is still the promoted Track2p-policy setting.
- Nearby IoU-distance thresholds act as deterministic stability votes.
- A bridge with insufficient support is removed by splitting; no alternative target is introduced.
- Splits that would create very short fragments are rejected by `--min-side-observations`.

Compare it against `track2p-policy` and `track2p-component-cleanup`. A useful result should reduce pairwise false positives without losing the complete-track advantage.
