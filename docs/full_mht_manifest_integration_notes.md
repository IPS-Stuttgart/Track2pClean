# Full MHT Manifest Integration, 2026-06-26

The frozen FullMHT manifests are now paired with benchmark-suite integration code.

## Added Artifacts

- `benchmarks/full_mht_prior_veto_manifest.json`
- `benchmarks/full_mht_prior_survival_sensitivity_manifest.json`
- `benchmarks/full_mht_terminal_completion_probe_manifest.json`
- `src/bayescatrack/experiments/_full_mht_manifest_integration.py`
- `src/bayescatrack/experiments/full_mht_manifest_decision.py`
- `tests/test_benchmark_manifest_full_mht_integration.py`
- `tests/test_full_mht_manifest_decision.py`
- `docs/full_mht_prior_survival_validation.md`
- `docs/full_mht_terminal_completion_objective.md`

## Runner Registration

The integration installs these manifest runner names:

| alias | canonical runner |
| --- | --- |
| `track2p-policy-full-mht` | `track2p-policy-full-mht` |
| `track2p-full-mht` | `track2p-policy-full-mht` |
| `track2p-pyrecest-full-mht` | `track2p-policy-full-mht` |

The adapter translates manifest fields into `FullMHTConfig` and keeps ordinary
Track2p `max_gap` separate from the FullMHT assignment/reactivation horizon,
which is named `full_mht_max_gap` in JSON manifests. Dynamic opt-in method fields
such as `track2p_prior_survival_*` and `terminal_incomplete_history_weight` are
attached to the frozen FullMHT config object before the run.

## Canonical Reproduction Manifest

`benchmarks/full_mht_prior_veto_manifest.json` compares:

| row | purpose |
| --- | --- |
| `Track2p` | original proposal baseline |
| `FullMHTPrior2` | full scan-assignment MHT with strong Track2p proposal prior |
| `FullMHTGreedyPrior2` | greedy beam-width-1 ablation over the same scan candidates and proposal-prior settings |
| `FullMHTPriorVetoScaled` | FullMHT with the fixed label-free prior-edge survival hazard |
| `FullMHTPriorSurvival` | FullMHT with calibrated label-free prior-edge survival likelihood |

The greedy row uses the same local scan candidate generator as `FullMHTPrior2`,
but sets `beam_width = 1`. It cannot preserve competing identity histories across
scans. If it ties the beam row on the real benchmark, the current data do not yet
demonstrate a history-search advantage. If it loses to the beam row, that is
direct evidence that full-history MHT is doing something local greedy assignment
cannot.

The prior-veto row freezes the first positive FullMHT-owned setting:

```text
track2p_prior_veto_penalty = 20.0
track2p_prior_veto_min_growth_residual_mahalanobis = 2.5
track2p_prior_veto_min_registered_iou = 0.35
track2p_prior_veto_max_registered_iou = 0.40
track2p_prior_veto_max_min_cell_probability = 0.65
```

The prior-survival row freezes the first calibrated survival candidate:

```text
track2p_prior_survival_weight = 1.0
track2p_prior_survival_min_examples_per_class = 2
track2p_prior_survival_score_clip = 8.0
```

After the canonical comparison CSV is produced, summarize the promotion gates
with:

```bash
"$PY" -m bayescatrack.experiments.full_mht_manifest_decision \
  "$OUT/full_mht_prior_veto/full_mht_prior_veto_comparison.csv" \
  --output "$OUT/full_mht_manifest_decision.md"
```

This decision artifact reports whether the full beam beats the greedy beam-width-1
ablation and whether calibrated prior-survival improves, ties, or falls below the
fixed prior-veto hazard.

## Sensitivity Manifests

`benchmarks/full_mht_prior_survival_sensitivity_manifest.json` checks the
immediate neighborhood around the calibrated survival row:

| factor | values |
| --- | --- |
| survival weight | `0.5`, `1.0`, `1.5` |
| survival score clip | `4.0`, `8.0` |
| minimum pseudo examples per class | `2`, `3` |
| anchor strictness | default vs stricter anchor overlap/confidence |

`benchmarks/full_mht_terminal_completion_probe_manifest.json` checks the immediate
neighborhood around the complete-history terminal objective:

| factor | values |
| --- | --- |
| terminal incomplete-history weight | `0.25`, `0.50`, `1.00` |

These are not tuning grids. They are robustness checks. If a method layer only
works at one point, keep it exploratory. If a small neighborhood is stable, the
layer can be considered for a future canonical row.

## Verification Status

The integration and validation artifacts were committed through the GitHub
connector because the local process runner was unavailable in the Codex desktop
environment. Static readback confirmed the committed files and branch diff, but
the Python tests and manifest executions still need to be run on the server
Python 3.12 environment.

Recommended focused check:

```bash
PY=/home/florianpfaff/codex-runs/BayesCaTrack/.venv312/bin/python
cd /home/florianpfaff/codex-runs/BayesCaTrack
export PYTHONPATH="$PWD/src"

"$PY" -m pytest -q \
  tests/test_benchmark_manifest_full_mht_integration.py \
  tests/test_full_mht_manifest_decision.py \
  tests/test_full_mht_prior_survival_model.py \
  tests/test_full_mht_prior_survival_integration.py \
  tests/test_full_mht_terminal_completion_integration.py \
  tests/test_track2p_policy_full_mht_conflict_demo.py \
  tests/test_track2p_policy_full_mht_growth_prior.py::test_full_mht_prior_veto_scoring_does_not_read_gt_audit_columns
```

Recommended manifest execution:

```bash
OUT="$PWD/results/full_mht_prior_survival_manifest_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
"$PY" -m bayescatrack benchmark suite \
  benchmarks/full_mht_prior_veto_manifest.json \
  --output-dir "$OUT" \
  --summary-format table

"$PY" -m bayescatrack.experiments.full_mht_manifest_decision \
  "$OUT/full_mht_prior_veto/full_mht_prior_veto_comparison.csv" \
  --output "$OUT/full_mht_manifest_decision.md"

SENS="$PWD/results/full_mht_prior_survival_sensitivity_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SENS"
"$PY" -m bayescatrack benchmark suite \
  benchmarks/full_mht_prior_survival_sensitivity_manifest.json \
  --output-dir "$SENS" \
  --summary-format table

COMP="$PWD/results/full_mht_terminal_completion_probe_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$COMP"
"$PY" -m bayescatrack benchmark suite \
  benchmarks/full_mht_terminal_completion_probe_manifest.json \
  --output-dir "$COMP" \
  --summary-format table
```

The full validation recipe, including the non-GT exposure audit, is in
`docs/full_mht_prior_survival_validation.md`. The terminal-completion probe is
described in `docs/full_mht_terminal_completion_objective.md`.
