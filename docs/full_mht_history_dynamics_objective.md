# FullMHT History Dynamics Objective, 2026-06-26

The FullMHT prototype now has two opt-in history-dynamics objectives. They are
not residual cleanup rules. They ask whether a seed-anchored identity history is
internally coherent under label-free registration, growth, and local-deformation
diagnostics.

## Motivation

Pairwise-good tracking can still be complete-track-bad. A single locally
plausible continuation can preserve pairwise scores while corrupting the full
identity history. FullMHT should therefore reason over histories, not just over
individual links.

## Terminal Reranking

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

The terminal selector uses:

```text
adjusted_score = scan_assignment_score
                 - terminal_history_risk_weight * prior_history_risk
                 - terminal_identity_history_risk
                 - terminal_motion_history_weight * motion_history_risk
```

With the default weight of `0.0`, base FullMHT behavior is unchanged.

## Scan-Time Beam Pruning

Terminal reranking is useful, but it can only choose among histories that survive
the beam. The stronger method layer is therefore scan-time pruning:

```python
from bayescatrack.experiments.full_mht_scan_history_dynamics_integration import (
    install_full_mht_scan_history_dynamics_pruning,
)

install_full_mht_scan_history_dynamics_pruning()
```

Configuration attribute:

```text
scan_motion_history_weight
```

This hook patches FullMHT beam pruning. After each scan, it parses the
label-free selected-edge summaries already produced by the FullMHT runner,
groups those edges back into partial identity histories, and subtracts a
within-history motion risk from the pruning score:

```text
beam_pruning_score = original_beam_pruning_score
                     - scan_motion_history_weight * partial_history_motion_risk
```

The risk includes both robust within-row outlier penalties and immediate
successive-edge deterioration penalties. The latter is important for short
partial histories: with only two selected edges, a median-only outlier score can
understate a bad second continuation, so the scan-time term also penalizes abrupt
IoU drops and abrupt growth/deformation residual jumps relative to the previous
edge.

This is the first layer in which MHT can preserve a globally cleaner identity
history even when its local scan score is slightly lower. It is still label-free:
it reads no manual-GT references, no benchmark scores, and no audit labels.

## Scan-History Conflict Demo

`full_mht_scan_history_conflict_demo` isolates the scan-pruning claim without any
benchmark scoring. It presents two complete candidate histories after the same
scans:

| candidate | local score | history behavior |
| --- | ---: | --- |
| `locally_high_score_motion_break` | higher | second edge has abrupt IoU collapse and growth/deformation jumps |
| `lower_score_motion_coherent` | lower | both selected edges remain internally coherent |

With `scan_motion_history_weight = 0.0`, the scan-history policy is identical to
local-score pruning and selects the higher raw-score candidate. With the frozen
demo weight of `1.0`, the label-free history risk flips the decision to the
coherent complete identity path. This is not real-data benchmark evidence; it is
an executable method invariant showing what the history-aware beam can do that a
local score cannot.

Run it with:

```bash
python -m bayescatrack.experiments.full_mht_scan_history_conflict_demo \
  --output results/full_mht_scan_history_conflict_selected.csv \
  --candidate-output results/full_mht_scan_history_conflict_candidates.csv
```

## Frozen Probe Manifests

The immediate terminal-weight neighborhood is frozen in:

```text
benchmarks/full_mht_history_dynamics_probe_manifest.json
```

Rows:

| row | purpose |
| --- | --- |
| `Track2p` | original proposal baseline |
| `FullMHTPrior2` | proposal-prior FullMHT control |
| `FullMHTHistoryDynamics025` | terminal motion-history weight `0.25` |
| `FullMHTHistoryDynamics050` | terminal motion-history weight `0.50` |
| `FullMHTHistoryDynamics100` | terminal motion-history weight `1.00` |

The immediate scan-pruning neighborhood is frozen in:

```text
benchmarks/full_mht_scan_history_dynamics_probe_manifest.json
```

Rows:

| row | purpose |
| --- | --- |
| `Track2p` | original proposal baseline |
| `FullMHTPrior2` | proposal-prior FullMHT control |
| `FullMHTScanHistoryDynamics025` | scan motion-history weight `0.25` |
| `FullMHTScanHistoryDynamics050` | scan motion-history weight `0.50` |
| `FullMHTScanHistoryDynamics100` | scan motion-history weight `1.00` |

Run both probes with:

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
  tests/test_full_mht_scan_history_dynamics_integration.py \
  tests/test_full_mht_scan_history_conflict_demo.py \
  tests/test_full_mht_history_dynamics_decision.py \
  tests/test_full_mht_scan_history_dynamics_decision.py \
  tests/test_full_mht_no_gt_leakage.py \
  tests/test_benchmark_manifest_full_mht_integration.py

"$PY" -m bayescatrack.experiments.full_mht_scan_history_conflict_demo \
  --output "$REPO/results/full_mht_scan_history_conflict_selected.csv" \
  --candidate-output "$REPO/results/full_mht_scan_history_conflict_candidates.csv"

TERMINAL_OUT="$REPO/results/full_mht_history_dynamics_probe_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TERMINAL_OUT"
"$PY" -m bayescatrack benchmark suite \
  benchmarks/full_mht_history_dynamics_probe_manifest.json \
  --output-dir "$TERMINAL_OUT" \
  --summary-format table

SCAN_OUT="$REPO/results/full_mht_scan_history_dynamics_probe_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SCAN_OUT"
"$PY" -m bayescatrack benchmark suite \
  benchmarks/full_mht_scan_history_dynamics_probe_manifest.json \
  --output-dir "$SCAN_OUT" \
  --summary-format table

"$PY" -m bayescatrack.experiments.full_mht_history_dynamics_decision \
  "$TERMINAL_OUT/full_mht_history_dynamics/full_mht_history_dynamics_comparison.csv" \
  --output "$TERMINAL_OUT/full_mht_history_dynamics_decision.md"

"$PY" -m bayescatrack.experiments.full_mht_scan_history_dynamics_decision \
  "$SCAN_OUT/full_mht_scan_history_dynamics/full_mht_scan_history_dynamics_comparison.csv" \
  --output "$SCAN_OUT/full_mht_scan_history_dynamics_decision.md"
```

## No-GT Leakage Guard

`tests/test_full_mht_no_gt_leakage.py` reads the method-layer source files and
fails if selector code references manual-GT or audit-result columns such as
`edge_status_against_gt`, `pairwise_delta_if_removed`, or
`complete_delta_if_removed`. The benchmark scorer still uses manual-GT to
measure final rows; the method layers must not use it for selection.

## Frozen Decision Rule

The decision helpers do not tune the method. They only interpret frozen
comparison tables:

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
