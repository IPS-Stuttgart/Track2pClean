# Track2p-policy minimum-threshold promotion

The Track2p-policy minimum-threshold benchmark is now the promoted
BayesCaTrack Track2p-style result row.  A DP rescue family was evaluated as a
possible follow-up, but it underperformed the plain policy row and is therefore
kept as an explicitly experimental diagnostic.

## Current result summary

| approach | pairwise F1 micro | complete-track F1 micro | interpretation |
| --- | ---: | ---: | --- |
| Track2p default | 0.965116 | 0.924370 | External baseline |
| Track2p-policy min | 0.961444 | 0.933333 | Promoted BayesCaTrack policy row |
| DP conservative gap | 0.869704 | 0.866142 | Best DP ablation, retained only as experimental |

The policy row finds more true pairwise links and more true complete tracks than
Track2p, while also adding extra pairwise false positives.  The next result
workstream should therefore be prune-only: identify and remove high-risk policy
false-positive edges without adding rescue edges.

## Benchmark policy

Use `track2p-policy` as the main BayesCaTrack row in guarded comparisons:

```bash
bayescatrack benchmark track2p-policy \
  --data ../benchmark-raw-suite2p-subjects \
  --reference ../benchmark-raw-suite2p-subjects \
  --reference-kind manual-gt \
  --threshold-method min \
  --iou-distance-threshold 12 \
  --format csv \
  --output results/track2p_policy.csv
```

Run DP only when an explicit experimental comparison is needed:

```bash
TRACK2P_INCLUDE_POLICY_DP_EXPERIMENT=true \
poetry run python .github/scripts/run_track2p_benchmark.py
```

## False-positive audit

The next tuning target is to prune policy false positives.  Generate an edge
ledger with:

```bash
bayescatrack benchmark track2p-policy-audit \
  --data ../benchmark-raw-suite2p-subjects \
  --reference ../benchmark-raw-suite2p-subjects \
  --reference-kind manual-gt \
  --threshold-method min \
  --iou-distance-threshold 12 \
  --output results/track2p_policy_edge_ledger.csv \
  --summary-output results/track2p_policy_edge_summary.csv
```

Use the ledger to design conservative post-filters.  Any proposed post-filter
should be validated fold-cleanly and should satisfy:

1. remove policy false-positive edges,
2. keep policy true-positive edges,
3. preserve the complete-track F1 advantage,
4. avoid DP-style rescue edges unless they pass a strict no-conflict rule.

## Implementation note

The benchmark manifest supports both `track2p-policy` and
`track2p-policy-dp`, but only the former is intended as a default benchmark row.
The DP runner remains useful for diagnostics and future redesigns that enforce
policy-preserving constraints.
