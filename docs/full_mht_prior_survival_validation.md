# FullMHT Prior-Survival Validation Bundle, 2026-06-26

`FullMHTPriorSurvival` is the first FullMHT row that replaces the fixed
hand-gated prior-veto pocket with a calibrated, label-free prior-edge survival
likelihood. It should not be promoted from candidate row to paper method until it
passes the checks below.

## Frozen Reproduction

Run the canonical manifest first:

```bash
REPO=/home/florianpfaff/codex-runs/BayesCaTrack
PY="$REPO/.venv312/bin/python"
cd "$REPO"
git fetch origin
git checkout codex/full-mht-prototype
git reset --hard origin/codex/full-mht-prototype
export PYTHONPATH="$REPO/src"

"$PY" -m pytest -q \
  tests/test_benchmark_manifest_full_mht_integration.py \
  tests/test_full_mht_prior_survival_model.py \
  tests/test_full_mht_prior_survival_integration.py \
  tests/test_track2p_policy_full_mht_conflict_demo.py \
  tests/test_track2p_policy_full_mht_growth_prior.py::test_full_mht_prior_veto_scoring_does_not_read_gt_audit_columns

OUT="$REPO/results/full_mht_prior_survival_manifest_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
"$PY" -m bayescatrack benchmark suite \
  benchmarks/full_mht_prior_veto_manifest.json \
  --output-dir "$OUT" \
  --summary-format table
```

Required rows:

| row | purpose |
| --- | --- |
| `Track2p` | original proposal baseline |
| `FullMHTPrior2` | FullMHT with strong Track2p proposal prior |
| `FullMHTPriorVetoScaled` | fixed hand-gated prior-survival hazard |
| `FullMHTPriorSurvival` | calibrated label-free prior-survival likelihood |

Promotion requires `FullMHTPriorSurvival` to match or improve
`FullMHTPriorVetoScaled` on complete-track F1 without losing pairwise F1 beyond a
single-edge-scale fluctuation. If it ties `FullMHTPrior2`, keep it as an
implemented model layer but not the headline row.

## Sensitivity Table

Run the frozen neighborhood manifest next:

```bash
OUT="$REPO/results/full_mht_prior_survival_sensitivity_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
"$PY" -m bayescatrack benchmark suite \
  benchmarks/full_mht_prior_survival_sensitivity_manifest.json \
  --output-dir "$OUT" \
  --summary-format table
```

The manifest varies only immediate method-neighborhood settings:

| factor | values |
| --- | --- |
| survival weight | `0.5`, `1.0`, `1.5` |
| survival score clip | `4.0`, `8.0` |
| minimum pseudo examples per class | `2`, `3` |
| anchor strictness | default vs stricter anchor overlap/confidence |

Decision rule:

- complete-track F1 should stay at least as high as `FullMHTPrior2` for nearby
  settings;
- pairwise F1 should not collapse in any immediate neighbor;
- if only one exact setting works, report the survival layer as exploratory;
- if a small plateau works, `FullMHTPriorSurvival` can replace the fixed
  prior-veto hazard as the stronger method row.

## Non-GT Exposure Audit

The exposure audit is deliberately not a manual-GT benchmark. It runs the same
FullMHT prior-survival configuration with Track2p output as the reference/seed
source so that all Track2p-style subjects can be inspected for broad firing.
The scoring numbers in this audit are not paper metrics; the summary counts are
the point.

```bash
AUDIT="$REPO/results/full_mht_prior_survival_exposure_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$AUDIT"

"$PY" - <<'PY' "$REPO" "$AUDIT"
from __future__ import annotations

import csv
import sys
from pathlib import Path

from bayescatrack.experiments.full_mht_prior_survival_integration import (
    install_full_mht_prior_survival_scoring,
)
from bayescatrack.experiments.track2p_benchmark import Track2pBenchmarkConfig
from bayescatrack.experiments.track2p_policy_full_mht_benchmark import (
    FullMHTConfig,
    _write_rows,
    run_track2p_policy_full_mht,
)

repo = Path(sys.argv[1])
out = Path(sys.argv[2])
data = repo / "results" / "policy_dp" / "data_lightweight"
install_full_mht_prior_survival_scoring()

mht_config = FullMHTConfig(
    seed_source="track2p-output",
    beam_width=8,
    scan_hypotheses=8,
    edge_top_k=4,
    identity_diverse_beam=True,
    miss_cost=2.0,
    max_gap=1,
    gap_reactivation_cost=1.0,
    min_output_observations=1,
    min_edge_score=0.25,
    track2p_prior_weight=12.0,
    track2p_non_prior_penalty=2.0,
    track2p_prior_switch_penalty=8.0,
    track2p_no_prior_successor_penalty=8.0,
    track2p_prior_miss_penalty=4.0,
)
object.__setattr__(mht_config, "track2p_prior_survival_weight", 1.0)
object.__setattr__(mht_config, "track2p_prior_survival_min_examples_per_class", 2)
object.__setattr__(mht_config, "track2p_prior_survival_score_clip", 8.0)

config = Track2pBenchmarkConfig(
    data=data,
    reference=data,
    reference_kind="track2p-output",
    method="global-assignment",
    input_format="suite2p",
    transform_type="affine",
    allow_track2p_as_reference_for_smoke_test=True,
    include_non_cells=False,
    cell_probability_threshold=0.5,
    exclude_overlapping_pixels=False,
    weighted_masks=False,
    weighted_centroids=False,
)

result = run_track2p_policy_full_mht(
    config,
    threshold_method="min",
    iou_distance_threshold=12.0,
    transform_type="affine",
    cell_probability_threshold=0.5,
    mht_config=mht_config,
    progress=True,
)

_write_rows(result.summary_rows, out / "summary.csv", output_format="csv")
_write_rows(result.diagnostic_rows, out / "diagnostics.csv", output_format="csv")
_write_rows(
    [benchmark_result.to_dict() for benchmark_result in result.results],
    out / "scores_against_track2p_reference.csv",
    output_format="csv",
)

with (out / "exposure_counts.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=(
            "subject",
            "scan_selected_prior_edges",
            "scan_selected_non_prior_edges",
            "scan_missed_prior_successors",
            "scan_switched_prior_successors",
            "scan_no_prior_successor_continuations",
            "terminal_selected_rank",
        ),
    )
    writer.writeheader()
    for row in result.summary_rows:
        if row.get("subject") == "ALL":
            continue
        writer.writerow(
            {
                "subject": row.get("subject", ""),
                "scan_selected_prior_edges": row.get("scan_selected_prior_edges", 0),
                "scan_selected_non_prior_edges": row.get(
                    "scan_selected_non_prior_edges", 0
                ),
                "scan_missed_prior_successors": row.get(
                    "scan_missed_prior_successors", 0
                ),
                "scan_switched_prior_successors": row.get(
                    "scan_switched_prior_successors", 0
                ),
                "scan_no_prior_successor_continuations": row.get(
                    "scan_no_prior_successor_continuations", 0
                ),
                "terminal_selected_rank": row.get("terminal_selected_rank", 1),
            }
        )

print(out)
PY
```

Decision rule:

- `scan_selected_non_prior_edges` should remain rare;
- `scan_missed_prior_successors` should remain tiny, not broad across subjects;
- no subject should receive a large number of prior switches or no-prior
  continuations;
- if exposure is broad, the survival model is too permissive or too strong even
  if the manual-GT benchmark improves.

## Recording

After the server runs, update this document and
`docs/full_mht_manifest_integration_notes.md` with:

- output directories;
- focused pytest result;
- canonical comparison table;
- sensitivity table;
- exposure counts table;
- promote / keep exploratory / reject decision.
