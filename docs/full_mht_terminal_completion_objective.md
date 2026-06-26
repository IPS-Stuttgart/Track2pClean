# FullMHT Terminal Completion Objective, 2026-06-26

The FullMHT prototype now has an opt-in complete-history terminal objective. It
is deliberately not enabled in the canonical prior-veto/prior-survival rows by
default.

## Motivation

The paper-facing method claim is that longitudinal tracking should choose whole
identity histories, not just locally plausible pairwise links. A local scan score
can prefer a continuation that looks good now but leaves an identity incomplete
or creates a worse terminal history. The terminal completion objective gives the
final MHT beam a label-free way to prefer more complete seed-anchored histories.

## Mechanism

Install hook:

```python
from bayescatrack.experiments.full_mht_terminal_completion_integration import (
    install_full_mht_terminal_completion_objective,
)

install_full_mht_terminal_completion_objective()
```

Configuration attribute:

```text
terminal_incomplete_history_weight
```

Risk definition:

```text
risk = terminal_incomplete_history_weight
       * count_missing_observations_in_non_empty_terminal_histories
```

Rows with no observations are ignored. Rows with at least one observation are
penalized for missing sessions. The term enters the existing terminal identity
history risk, so it can affect both beam pruning and final hypothesis selection
once installed. With the default weight of `0.0`, base FullMHT behavior is
unchanged.

## Frozen Probe Manifest

The immediate weight neighborhood is frozen in:

```text
benchmarks/full_mht_terminal_completion_probe_manifest.json
```

Rows:

| row | purpose |
| --- | --- |
| `Track2p` | original proposal baseline |
| `FullMHTPrior2` | proposal-prior FullMHT control |
| `FullMHTTerminalCompletion025` | terminal completion weight `0.25` |
| `FullMHTTerminalCompletion050` | terminal completion weight `0.50` |
| `FullMHTTerminalCompletion100` | terminal completion weight `1.00` |

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
  tests/test_full_mht_terminal_completion_integration.py \
  tests/test_benchmark_manifest_full_mht_integration.py

OUT="$REPO/results/full_mht_terminal_completion_probe_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
"$PY" -m bayescatrack benchmark suite \
  benchmarks/full_mht_terminal_completion_probe_manifest.json \
  --output-dir "$OUT" \
  --summary-format table
```

## Direct Probe Runner

The wrapper below delegates to the base FullMHT runner and only adds the terminal
completion attribute. Use it when diagnostics or a one-off weight are needed:

```bash
DIRECT="$REPO/results/full_mht_terminal_completion_direct_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DIRECT"

"$PY" -m bayescatrack.experiments.track2p_policy_full_mht_terminal_completion_benchmark \
  --data "$REPO/results/policy_dp/data_lightweight" \
  --reference "$REPO/results/policy_dp/data_lightweight" \
  --reference-kind manual-gt \
  --input-format suite2p \
  --threshold-method min \
  --transform-type affine \
  --iou-distance-threshold 12 \
  --cell-probability-threshold 0.5 \
  --seed-source reference \
  --beam-width 8 \
  --scan-hypotheses 8 \
  --edge-top-k 4 \
  --identity-diverse-beam \
  --miss-cost 2.0 \
  --max-gap 1 \
  --gap-reactivation-cost 1.0 \
  --min-output-observations 1 \
  --min-edge-score 0.25 \
  --track2p-prior-weight 12.0 \
  --track2p-non-prior-penalty 2.0 \
  --track2p-prior-switch-penalty 8.0 \
  --track2p-no-prior-successor-penalty 8.0 \
  --track2p-prior-miss-penalty 4.0 \
  --terminal-incomplete-history-weight 0.5 \
  --output "$DIRECT/full_mht_terminal_completion.csv" \
  --format csv \
  --diagnostics-output "$DIRECT/diagnostics.csv" \
  --diagnostics-format csv \
  --summary-output "$DIRECT/summary.csv" \
  --progress
```

## Interpretation

This is not yet a promoted row. It is a method probe.

Positive evidence would be:

- pairwise F1 stays close to `FullMHTPrior2`;
- complete-track F1 improves over `FullMHTPrior2` or the greedy beam row;
- diagnostics show reranking toward complete histories without broad non-prior
  continuations or prior switches;
- at least two nearby weights are stable, not a single exact spike.

Negative evidence would be:

- the term simply rewards over-linking and damages pairwise F1;
- it has no effect because the beam never preserves the relevant alternatives;
- it only works at one fragile weight.

If promising, add only the stable terminal-completion row to the canonical
manifest after recording the probe output directory and comparison table.
