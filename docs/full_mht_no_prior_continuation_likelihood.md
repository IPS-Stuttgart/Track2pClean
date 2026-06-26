# FullMHT No-Prior Continuation Likelihood, 2026-06-26

This method layer targets the main failure mode of the calibrated association
FullMHT probe: unsafe non-prior continuation chains. Those chains occur when
Track2p has no proposal successor for a source ROI, but the local calibrated edge
score still makes a continuation look attractive.

The earlier scalar control was:

```text
track2p_no_prior_successor_penalty
```

It worked only brittlely: small penalties admitted many unsafe continuations,
while slightly larger penalties collapsed the row back to the conservative
proposal-control solution. The new layer replaces that scalar-only behavior with
a label-free likelihood ratio:

```text
log p(continue despite no Track2p successor | label-free diagnostics)
  - log p(death / no continuation | label-free diagnostics)
```

## Method Layer

The hook applies only to candidate edges that satisfy all conditions below:

```text
edge is not a Track2p proposal edge
source ROI has no Track2p proposal successor in the current scan
no_prior_continuation_likelihood_weight > 0
```

It builds pseudo-continuation anchors from high-confidence local geometry:

- high registered and shifted IoU;
- low growth residual and Mahalanobis residual;
- high endpoint cell probability;
- low local-neighborhood deformation;
- top row/column ranks.

Weak candidates form the death/background reference distribution. A robust
Gaussian likelihood ratio then adds positive score to anchor-like no-prior
continuations and negative score to weak no-prior continuations. The hook is
installed through manifest support only when a `no_prior_continuation_*` option is
present.

## Frozen Probe

The immediate weight neighborhood is frozen in:

```text
benchmarks/full_mht_no_prior_continuation_probe_manifest.json
```

Rows:

| row | purpose |
| --- | --- |
| `Track2p` | original proposal baseline |
| `FullMHTPrior2` | proposal-prior FullMHT control |
| `FullMHTCalibratedNoDeath` | calibrated association with scalar no-prior penalty disabled |
| `FullMHTNoPriorContinuation050` | no-prior likelihood weight `0.5` |
| `FullMHTNoPriorContinuation100` | no-prior likelihood weight `1.0` |
| `FullMHTNoPriorContinuation150` | no-prior likelihood weight `1.5` |

All no-prior likelihood rows keep:

```text
association_score_mode = calibrated-likelihood
association_likelihood_clip = 4.0
track2p_no_prior_successor_penalty = 0.0
no_prior_continuation_min_examples_per_class = 2
no_prior_continuation_score_clip = 8.0
```

## Validation Command

```bash
REPO=/home/florianpfaff/codex-runs/BayesCaTrack
PY="$REPO/.venv312/bin/python"
cd "$REPO"
git fetch origin
git checkout codex/full-mht-prototype
git reset --hard origin/codex/full-mht-prototype
export PYTHONPATH="$REPO/src"

"$PY" -m pytest -q \
  tests/test_full_mht_no_prior_continuation_model.py \
  tests/test_full_mht_no_prior_continuation_integration.py \
  tests/test_full_mht_no_prior_continuation_manifest_integration.py \
  tests/test_full_mht_no_prior_continuation_decision.py \
  tests/test_full_mht_no_gt_leakage.py

OUT="$REPO/results/full_mht_no_prior_continuation_probe_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
"$PY" -m bayescatrack benchmark suite \
  benchmarks/full_mht_no_prior_continuation_probe_manifest.json \
  --output-dir "$OUT" \
  --summary-format table

"$PY" -m bayescatrack.experiments.full_mht_no_prior_continuation_decision \
  "$OUT/full_mht_no_prior_continuation/full_mht_no_prior_continuation_comparison.csv" \
  --output "$OUT/full_mht_no_prior_continuation_decision.md"
```

## Decision Rule

Treat this as a method probe until the manifest shows:

- pairwise F1 does not regress against `FullMHTPrior2`;
- complete-track F1 improves or at least returns to `FullMHTPrior2`;
- no-prior continuations are reduced relative to `FullMHTCalibratedNoDeath`;
- the effect is not limited to a single weight.

The decision helper freezes the metric part of this rule as
`no_prior_continuation_stable_gain`, `no_prior_continuation_single_weight_gain`,
`no_prior_continuation_ties_baseline`, `no_prior_continuation_pairwise_regression`,
or `no_prior_continuation_complete_regression`. It does not replace the exposure
audit; the final judgment still has to inspect whether selected no-prior
continuations remain rare.

If it improves complete-track identity across nearby weights, it is stronger
method evidence than the scalar death penalty. If it collapses to `FullMHTPrior2`,
record it as a useful label-free model layer but not a promoted row. If it still
admits broad no-prior chains, the next step should combine this likelihood with
history-diverse beam retention and growth-history prediction rather than tuning a
single scalar.
