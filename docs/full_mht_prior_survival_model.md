# FullMHT Prior-Edge Survival Model, 2026-06-26

The branch now contains a label-free calibrated survival model for Track2p prior
edges and an opt-in FullMHT scoring integration:

```text
src/bayescatrack/experiments/full_mht_prior_survival_model.py
src/bayescatrack/experiments/full_mht_prior_survival_integration.py
```

This is the intended replacement path for the fixed prior-veto hazard pocket.
Instead of saying that one hand-gated pocket is suspicious, FullMHT can now add a
calibrated survival log-likelihood ratio to Track2p proposal edges during
scan-assignment history selection.

## Why This Exists

The positive `FullMHTPriorVetoScaled` row is methodologically better than
post-hoc cleanup because the suspicious prior edge is penalized during full
history selection. But it still uses a hand-gated pocket. A paper-facing Bayesian
method should instead assign a survival likelihood to each Track2p proposal edge:

```text
log p(prior edge survives | label-free diagnostics)
```

The new model and integration implement that calibration layer without reading
manual-GT labels or audit-result columns.

## Features

Each prior edge is represented by:

| feature | orientation |
| --- | --- |
| registered IoU | higher means more survival-like |
| shifted IoU | higher means more survival-like |
| minimum endpoint cell probability | higher means more survival-like |
| area similarity | closer to 1 means more survival-like |
| negative growth residual | higher means lower residual |
| negative growth Mahalanobis | higher means lower growth surprise |
| negative local deformation | higher means locally coherent |
| negative log row rank | higher means better row rank |
| negative log column rank | higher means better column rank |
| terminal-edge indicator | learned from pseudo distributions |
| last-session indicator | learned from pseudo distributions |
| complete-component indicator | learned from pseudo distributions |

## Calibration

The model creates two label-free pseudo classes:

- high-confidence survival anchors: high registered/shifted overlap, low growth
  residual, high endpoint cell probability, and rank-1 local support;
- risky prior-edge background examples: low overlap, high growth residual, weak
  endpoint confidence, or poor local support.

It then fits robust per-feature Gaussian location/scale summaries and scores each
edge by a clipped log-likelihood ratio. The score is positive for survival-like
prior edges and negative for hazard-like prior edges.

## FullMHT Integration

`install_full_mht_prior_survival_scoring()` patches FullMHT edge scoring in an
idempotent, opt-in way. The baseline score is unchanged unless the active
`FullMHTConfig` has a nonzero dynamic attribute:

```text
track2p_prior_survival_weight
```

When enabled, only Track2p proposal edges receive the survival term. Non-prior
candidate edges keep the existing association score and prior-switch penalties.
This keeps the method focused on the Bayesian question: should a proposed prior
identity edge survive in the full scan-assignment history?

Manifest rows can now set the survival controls directly. The manifest adapter
installs the scoring integration automatically when any
`track2p_prior_survival_*` field is present, attaches the values to the frozen
`FullMHTConfig`, and runs the usual `track2p-policy-full-mht` runner.

Example knobs:

```json
{
  "runner": "track2p-full-mht",
  "track2p_prior_weight": 12.0,
  "track2p_prior_survival_weight": 1.0,
  "track2p_prior_survival_min_anchor_registered_iou": 0.75,
  "track2p_prior_survival_min_anchor_shifted_iou": 0.65,
  "track2p_prior_survival_min_examples_per_class": 2,
  "track2p_prior_survival_score_clip": 8.0
}
```

## Tests

The committed tests are:

```text
tests/test_full_mht_prior_survival_model.py
tests/test_full_mht_prior_survival_integration.py
tests/test_benchmark_manifest_full_mht_integration.py
```

They check that the model:

- separates anchor-like prior edges from suspicious prior edges;
- uses label-free pseudo masks;
- falls back to zero when there is not enough pseudo-label support;
- orients growth/rank features as survival evidence;
- clips extreme scores;
- changes FullMHT scoring only when the survival weight is enabled;
- is installed idempotently;
- can be configured from a benchmark manifest.

## Verification Still Needed

The integration is now code-complete enough to run a frozen manifest row, but it
has not yet been benchmarked on the server in this Codex session. The promotion
condition is not merely that this module exists. It must improve or match
`FullMHTPriorVetoScaled` under the frozen manifest while showing a stable
sensitivity/exposure profile.

Run the focused tests on the Python 3.12 benchmark server:

```bash
REPO=/home/florianpfaff/codex-runs/BayesCaTrack
PY="$REPO/.venv312/bin/python"
cd "$REPO"
git fetch origin
git checkout codex/full-mht-prototype
git reset --hard origin/codex/full-mht-prototype
export PYTHONPATH="$REPO/src"

"$PY" -m pytest -q \
  tests/test_full_mht_prior_survival_model.py \
  tests/test_full_mht_prior_survival_integration.py \
  tests/test_benchmark_manifest_full_mht_integration.py
```

Then run a manifest row that compares at least:

```text
Track2p
FullMHTPrior2
FullMHTPriorVetoScaled
FullMHTPriorSurvival
```

If the survival row cannot match the fixed prior-veto row, keep it as a method
layer but do not promote it as the paper-facing result yet.
