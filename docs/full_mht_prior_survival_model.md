# FullMHT Prior-Edge Survival Model, 2026-06-26

The branch now contains a standalone label-free calibrated survival model for
Track2p prior edges:

```text
src/bayescatrack/experiments/full_mht_prior_survival_model.py
```

This is the intended replacement for the fixed prior-veto hazard pocket once it
is wired into the FullMHT scan-assignment score.

## Why This Exists

The positive `FullMHTPriorVetoScaled` row is methodologically better than
post-hoc cleanup because the suspicious prior edge is penalized during full
history selection. But it still uses a hand-gated pocket. A paper-facing Bayesian
method should instead assign a survival likelihood to each Track2p proposal edge:

```text
log p(prior edge survives | label-free diagnostics)
```

The new module implements that calibration layer without reading manual-GT labels
or audit-result columns.

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
- risky prior-edge background examples: low overlap, high growth residual,
  weak endpoint confidence, or poor local support.

It then fits robust per-feature Gaussian location/scale summaries and scores each
edge by a clipped log-likelihood ratio. The score is positive for survival-like
prior edges and negative for hazard-like prior edges.

## Tests

The committed tests are:

```text
tests/test_full_mht_prior_survival_model.py
```

They check that the model:

- separates anchor-like prior edges from suspicious prior edges;
- uses label-free pseudo masks;
- falls back to zero when there is not enough pseudo-label support;
- orients growth/rank features as survival evidence;
- clips extreme scores.

## Integration Plan

The next code step is to wire this into `track2p_policy_full_mht_benchmark.py`:

1. collect `PriorEdgeSurvivalDiagnostics` for Track2p prior candidates within a
   subject or scan;
2. calibrate `PriorEdgeSurvivalModel` from label-free pseudo anchors/background;
3. add the model's log survival ratio to Track2p prior-edge scoring;
4. preserve the current fixed `track2p_prior_veto_penalty` path as a baseline;
5. report survival scores in `scan_selected_edge_summaries` and summary rows.

The promotion condition is not merely that this module exists. It must improve or
match `FullMHTPriorVetoScaled` under the frozen manifest while showing a stable
sensitivity/exposure profile.

## Verification Still Needed

The local Codex process runner was unavailable when this module was committed, so
run the tests on the Python 3.12 benchmark server:

```bash
REPO=/home/florianpfaff/codex-runs/BayesCaTrack
PY="$REPO/.venv312/bin/python"
cd "$REPO"
git fetch origin
git checkout codex/full-mht-prototype
git reset --hard origin/codex/full-mht-prototype
export PYTHONPATH="$REPO/src"

"$PY" -m pytest -q tests/test_full_mht_prior_survival_model.py
```
