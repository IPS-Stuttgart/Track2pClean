# FullMHT Identity-History Consistency Likelihood

The FullMHT prototype can now be run with an experimental identity-history consistency score.  This is meant to test the method jump that the paper needs: candidate continuations should be judged not only by the current session pair, but also by whether they are plausible for the identity history carried by the current MHT hypothesis.

## Method Rationale

The base FullMHT row scores candidate scan assignments from local features such as registered IoU, shifted IoU, cell probability, growth residual, local deformation, and optional Track2p-prior terms.  That is useful, but it can still accept false continuations that look locally plausible.

The history-consistency variant adds a per-track, label-free likelihood layer.  For a candidate edge, it reconstructs the same track row's previous accepted consecutive edges and compares the candidate against that identity-specific history using:

| feature | risky direction |
| --- | --- |
| registered IoU | lower than history |
| shifted IoU | lower than history |
| minimum endpoint cell probability | lower than history |
| growth residual | higher than history |
| growth Mahalanobis residual | higher than history |
| local deformation | higher than history |

Risk is charged only when both groups agree: weak overlap/cell evidence and high growth/motion/deformation evidence.  A single weak feature is intentionally not enough to break a history.

## Experimental Runner

This is currently a module-level wrapper around the base runner:

```bash
python -m bayescatrack.experiments.track2p_policy_full_mht_history_consistency_benchmark \
  --history-consistency-weight 1.0 \
  --history-consistency-min-history-edges 2 \
  --history-consistency-joint-margin 1.0 \
  --data "$DATA" \
  --reference "$REF" \
  --reference-kind manual-gt \
  --input-format suite2p \
  --threshold-method min \
  --transform-type affine \
  --iou-distance-threshold 12 \
  --cell-probability-threshold 0.5 \
  --seed-source track2p-output \
  --track2p-prior-weight 12 \
  --track2p-non-prior-penalty 2 \
  --track2p-prior-switch-penalty 8 \
  --track2p-no-prior-successor-penalty 8 \
  --track2p-prior-miss-penalty 4 \
  --beam-width 4 \
  --scan-hypotheses 4 \
  --edge-top-k 4 \
  --miss-cost 2 \
  --output "$OUT/full_mht_history_consistency.csv" \
  --format csv \
  --diagnostics-output "$OUT/full_mht_history_consistency_diagnostics.csv" \
  --diagnostics-format csv \
  --summary-output "$OUT/full_mht_history_consistency_summary.csv" \
  --progress
```

The wrapper strips the `--history-consistency-*` options, patches the base FullMHT scan-expansion function during the run, and restores the base runner afterward.

## Decision Use

This row should not be promoted just because it moves one known edge.  It is useful only if it shows a stable pattern that the MHT beam is selecting better complete identity histories than the local FullMHT or Track2p-prior replay row.

Promote only if a frozen setting:

- improves complete-track micro F1 over Track2p-prior replay and ideally over CoherenceSuffixGrowthVeto;
- does not reduce pairwise micro F1 below the accepted non-teacher row;
- does not create broad missed-track collapse;
- shows selected history penalties are rare and concentrated on identity-history outliers;
- survives a small sensitivity table over weight and joint margin.

If it simply ties Track2p-prior replay, record it as evidence that identity-history consistency is not selective enough yet.  If it collapses, the likely failure mode is over-penalizing real morphology changes after otherwise stable histories.
