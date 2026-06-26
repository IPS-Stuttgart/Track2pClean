# Full MHT Manifest Integration, 2026-06-26

The frozen FullMHT prior-veto manifest is now paired with benchmark-suite
integration code.

## Added Artifacts

- `benchmarks/full_mht_prior_veto_manifest.json`
- `src/bayescatrack/experiments/_full_mht_manifest_integration.py`
- `tests/test_benchmark_manifest_full_mht_integration.py`

## Runner Registration

The integration installs these manifest runner names:

| alias | canonical runner |
| --- | --- |
| `track2p-policy-full-mht` | `track2p-policy-full-mht` |
| `track2p-full-mht` | `track2p-policy-full-mht` |
| `track2p-pyrecest-full-mht` | `track2p-policy-full-mht` |

The adapter translates manifest fields into `FullMHTConfig` and keeps ordinary
Track2p `max_gap` separate from the FullMHT assignment/reactivation horizon,
which is named `full_mht_max_gap` in JSON manifests.

## Method Row Frozen By The Manifest

The manifest compares:

| row | purpose |
| --- | --- |
| `Track2p` | original proposal baseline |
| `FullMHTPrior2` | full scan-assignment MHT with strong Track2p proposal prior |
| `FullMHTPriorVetoScaled` | FullMHT with the label-free prior-edge survival hazard |

The prior-veto row freezes the first positive FullMHT-owned setting:

```text
track2p_prior_veto_penalty = 20.0
track2p_prior_veto_min_growth_residual_mahalanobis = 2.5
track2p_prior_veto_min_registered_iou = 0.35
track2p_prior_veto_max_registered_iou = 0.40
track2p_prior_veto_max_min_cell_probability = 0.65
```

## Verification Status

The integration was committed through the GitHub connector because the local
process runner was unavailable in the Codex desktop environment. Static readback
confirmed the committed files and branch diff, but the Python tests and the full
manifest execution still need to be run on the server Python 3.12 environment.

Recommended focused check:

```bash
PY=/home/florianpfaff/codex-runs/BayesCaTrack/.venv312/bin/python
cd /home/florianpfaff/codex-runs/BayesCaTrack
export PYTHONPATH="$PWD/src"

"$PY" -m pytest -q \
  tests/test_benchmark_manifest_full_mht_integration.py \
  tests/test_track2p_policy_full_mht_growth_prior.py::test_full_mht_prior_veto_scoring_does_not_read_gt_audit_columns
```

Recommended manifest execution:

```bash
OUT="$PWD/results/full_mht_prior_veto_manifest_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
"$PY" -m bayescatrack benchmark suite \
  benchmarks/full_mht_prior_veto_manifest.json \
  --output-dir "$OUT" \
  --summary-format table
```
