# Full MHT Manifest Integration, 2026-06-26

The frozen FullMHT manifests are now paired with benchmark-suite integration code.

## Added Artifacts

- `benchmarks/full_mht_prior_veto_manifest.json`
- `benchmarks/full_mht_prior_survival_sensitivity_manifest.json`
- `benchmarks/full_mht_terminal_completion_probe_manifest.json`
- `src/bayescatrack/experiments/_full_mht_manifest_integration.py`
- `src/bayescatrack/experiments/full_mht_manifest_decision.py`
- `src/bayescatrack/experiments/full_mht_prior_survival_promotion_gate.py`
- `src/bayescatrack/experiments/track2p_policy_full_mht_exposure_audit.py`
- `tests/test_benchmark_manifest_full_mht_integration.py`
- `tests/test_full_mht_manifest_decision.py`
- `tests/test_full_mht_prior_survival_promotion_gate.py`
- `tests/test_full_mht_exposure_audit.py`
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
| `FullMHTGreedyPrior2` | greedy beam-width-1 ablation for the proposal-prior control |
| `FullMHTPriorVetoScaled` | FullMHT with the fixed label-free prior-edge survival hazard |
| `FullMHTGreedyPriorVetoScaled` | greedy beam-width-1 ablation for the fixed hazard |
| `FullMHTPriorSurvival` | FullMHT with calibrated label-free prior-edge survival likelihood |
| `FullMHTGreedyPriorSurvival` | greedy beam-width-1 ablation for the calibrated survival row |

Each greedy row uses the same local scan candidate generator and scoring options
as its beam counterpart, but sets `beam_width = 1` and disables identity-diverse
beam retention. If a candidate row ties its greedy ablation on the real benchmark,
the current data do not yet demonstrate a candidate-specific history-search
advantage. If the candidate beam wins on complete-track F1 without pairwise loss,
that is direct evidence that full-history MHT is doing something local greedy
assignment cannot.

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

After the canonical comparison CSV is produced, summarize the manifest decision
with:

```bash
"$PY" -m bayescatrack.experiments.full_mht_manifest_decision \
  "$OUT/full_mht_prior_veto/full_mht_prior_veto_comparison.csv" \
  --output "$OUT/full_mht_manifest_decision.md"
```

This decision artifact reports whether the base beam, fixed-veto beam, and
calibrated-survival beam beat their matching greedy beam-width-1 ablations, and
whether calibrated prior-survival improves, ties, or falls below the fixed
prior-veto hazard.

## Sensitivity And Exposure

`benchmarks/full_mht_prior_survival_sensitivity_manifest.json` checks the
immediate neighborhood around the calibrated survival row:

| factor | values |
| --- | --- |
| survival weight | `0.5`, `1.0`, `1.5` |
| survival score clip | `4.0`, `8.0` |
| minimum pseudo examples per class | `2`, `3` |
| anchor strictness | default vs stricter anchor overlap/confidence |

`track2p_policy_full_mht_exposure_audit.py` runs a label-free audit across
Track2p-style subjects and now accepts the same prior-survival scoring flags. The
combined promotion helper is:

```bash
"$PY" -m bayescatrack.experiments.full_mht_prior_survival_promotion_gate \
  "$OUT/full_mht_prior_veto/full_mht_prior_veto_comparison.csv" \
  "$SENS/full_mht_prior_survival_sensitivity/full_mht_prior_survival_sensitivity.csv" \
  "$AUDIT/prior_survival_exposure.csv" \
  --output "$AUDIT/prior_survival_promotion_gate.md"
```

It requires candidate-specific complete-history beam advantage, stable sensitivity,
bounded label-free exposure, and active prior-survival scoring. If any gate fails,
keep the calibrated survival row exploratory.

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
  tests/test_full_mht_prior_survival_promotion_gate.py \
  tests/test_full_mht_exposure_audit.py \
  tests/test_full_mht_no_gt_leakage.py \
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
```

The full validation recipe, including the exact non-GT exposure audit command and
combined promotion gate, is in `docs/full_mht_prior_survival_validation.md`. The
terminal-completion probe is described in `docs/full_mht_terminal_completion_objective.md`.
