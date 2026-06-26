# FullMHT History Dynamics Objective, 2026-06-26

The FullMHT prototype now has an opt-in terminal history-dynamics objective. It
is not a residual cleanup rule. It asks whether a final seed-anchored identity
history contains an edge that is an outlier relative to the other edges in the
same history.

## Motivation

Pairwise-good tracking can still be complete-track-bad. A single locally
plausible continuation can preserve pairwise scores while corrupting the full
identity history. This objective gives FullMHT a label-free terminal reranking
term for that failure mode: prefer the hypothesis whose whole history is more
internally coherent.

## Mechanism

Install hook:

```python
from bayescatrack.experiments.full_mht_history_dynamics_integration import (
    install_full_mht_history_dynamics_objective,
)

install_full_mht_history_dynamics_objective()
```

Configuration attribute:

```text
terminal_motion_history_weight
```

For every terminal hypothesis, each observed track row is decomposed into its
selected edges. Edges in rows with at least three observations are compared
against the other edges in that same history using label-free diagnostics already
computed by FullMHT:

- registered IoU;
- shifted IoU;
- growth residual;
- growth Mahalanobis residual;
- local-neighborhood deformation.

The risk is a robust within-history outlier penalty. Low IoU outliers, high
growth-residual outliers, high Mahalanobis outliers, high local-deformation
outliers, and missing edge diagnostics increase risk. The terminal selector uses:

```text
adjusted_score = scan_assignment_score
                 - terminal_history_risk_weight * prior_history_risk
                 - terminal_identity_history_risk
                 - terminal_motion_history_weight * motion_history_risk
```

With the default weight of `0.0`, base FullMHT behavior is unchanged.

## Frozen Probe Manifest

The immediate weight neighborhood is frozen in:

```text
benchmarks/full_mht_history_dynamics_probe_manifest.json
```

Rows:

| row | purpose |
| --- | --- |
| `Track2p` | original proposal baseline |
| `FullMHTPrior2` | proposal-prior FullMHT control |
| `FullMHTHistoryDynamics025` | motion-history weight `0.25` |
| `FullMHTHistoryDynamics050` | motion-history weight `0.50` |
| `FullMHTHistoryDynamics100` | motion-history weight `1.00` |

Run it with:

```bash
REPO=/home/florianpfaff/codex-runs/BayesCaTrack
PY="$REPO/.venv312/bin/python"
cd "$REPO"
git fetch origin
git checkout codex/full-mht-prototype
git reset --hard origin/codex/full-mht-prototype
export PYTHONPATH="$REPO/src"

"$PY" -m pytest -q \
  tests/test_full_mht_history_dynamics_integration.py \
  tests/test_full_mht_history_dynamics_decision.py \
  tests/test_benchmark_manifest_full_mht_integration.py

OUT="$REPO/results/full_mht_history_dynamics_probe_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
"$PY" -m bayescatrack benchmark suite \
  benchmarks/full_mht_history_dynamics_probe_manifest.json \
  --output-dir "$OUT" \
  --summary-format table

"$PY" -m bayescatrack.experiments.full_mht_history_dynamics_decision \
  "$OUT/full_mht_history_dynamics/full_mht_history_dynamics_comparison.csv" \
  --output "$OUT/full_mht_history_dynamics_decision.md"
```

## Frozen Decision Rule

The decision helper does not tune the method. It only interprets the frozen
comparison table:

| result | meaning |
| --- | --- |
| `history_dynamics_stable_gain` | at least two nearby weights improve complete-track F1 without pairwise regression |
| `history_dynamics_single_weight_gain` | only one weight improves complete-track F1, so treat it as knife-edge |
| `history_dynamics_ties_baseline` | history dynamics validates the story but does not improve metrics |
| `history_dynamics_pairwise_regression` | do not promote; history pressure damages pairwise tracking |
| `history_dynamics_complete_regression` | do not promote; complete-track identity is worse |

Promotion still requires the broader no-GT, exposure, and sensitivity gates.

## Interpretation

Positive evidence would be:

- pairwise F1 does not regress against `FullMHTPrior2`;
- complete-track F1 improves at more than one nearby weight;
- selected histories show fewer internally anomalous edges, not broad over-linking;
- the beam result beats or meaningfully differs from the greedy beam ablation.

Negative evidence would be:

- the term has no effect because the beam did not preserve the relevant
  alternatives;
- it damages pairwise F1 by over-preferring complete-looking histories;
- it only works at one exact weight.

If it works, this is a stronger method story than post-hoc cleanup: MHT is
choosing among full identity histories using label-free within-history dynamics.