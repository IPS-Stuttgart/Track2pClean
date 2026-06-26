# FullMHT Prior-Survival Diagnostics, 2026-06-26

The calibrated prior-edge survival integration now surfaces its contribution in
selected-edge summaries. When `track2p_prior_survival_weight` is nonzero and the
pseudo-label model has enough support, Track2p prior edges include:

```text
survival=<raw log survival ratio>
survival_weighted=<weight * raw log survival ratio>
```

These fields are appended inside the semicolon-separated
`selected_edge_summaries` strings written to the FullMHT diagnostic and summary
outputs.

## Interpretation

| field | meaning |
| --- | --- |
| `survival` | calibrated label-free log-likelihood ratio for prior-edge survival |
| `survival_weighted` | actual contribution added to the FullMHT edge score |

Positive values mean the edge looks survival-like under the pseudo-calibrated
model. Negative values mean the edge looks suspicious and receives a lower
survival likelihood during scan-assignment history selection.

If the model lacks enough pseudo-survival and pseudo-hazard support for a scan,
the summary appends:

```text
survival=disabled
```

That is important for the sensitivity/exposure audit: a row that appears stable
because the survival model often disables itself should be treated as an
implemented architecture layer, not as a proven calibrated method.

## Audit Questions

When reading the direct diagnostic run from
`docs/full_mht_prior_survival_validation.md`, check:

- Are negative `survival` values rare and concentrated on a tiny number of risky
  Track2p prior edges?
- Does `survival_weighted` explain any missed prior successors or terminal rank
  changes?
- Do high-confidence prior edges mostly receive positive or near-zero survival
  support?
- Does the exposure audit show broad negative survival pressure across subjects?

Promotion requires the survival term to behave like a selective prior-edge
likelihood, not a broad penalty that happens to improve one manual-GT benchmark.
