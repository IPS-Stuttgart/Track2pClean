"""Generate and run the guarded Track2p benchmark suite for GitHub Actions."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bayescatrack.dependency_pins import PYRECEST_COMMIT, PYRECEST_REPOSITORY
from bayescatrack.experiments.benchmark_manifest import (
    load_benchmark_manifest,
    run_benchmark_manifest,
)
from bayescatrack.experiments.track2p_benchmark import discover_subject_dirs


def _bool_env(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def _int_env(name: str, *, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return int(value)


def _json_object_env(name: str) -> dict[str, Any]:
    value = os.environ.get(name, "{}").strip() or "{}"
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must decode to a JSON object")
    return parsed


def _should_run_loso(policy: str, *, n_subjects: int) -> bool:
    normalized = policy.strip().casefold()
    if normalized == "auto":
        return n_subjects >= 2
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Unsupported calibrated LOSO policy: {policy!r}")


def _shifted_pairwise_kwargs(
    base_pairwise_cost_kwargs: Mapping[str, Any],
    *,
    radius: int,
    shift_penalty_weight: float,
    include_mask_cosine_flag: bool = False,
) -> dict[str, Any]:
    """Return pairwise kwargs for one shifted-overlap benchmark variant."""

    radius = int(radius)
    if radius < 0:
        raise ValueError("shifted-IoU benchmark radius must be non-negative")
    shift_penalty_weight = float(shift_penalty_weight)
    if shift_penalty_weight < 0.0:
        raise ValueError("shifted-IoU shift penalty weight must be non-negative")

    use_shifted_overlap = radius > 0
    kwargs = dict(base_pairwise_cost_kwargs)
    kwargs.update(
        {
            "shifted_iou_radius": radius,
            "use_shifted_iou_for_iou_cost": use_shifted_overlap,
            "shifted_iou_shift_penalty_weight": shift_penalty_weight,
        }
    )
    if include_mask_cosine_flag:
        kwargs["use_shifted_mask_cosine_for_mask_cosine_cost"] = use_shifted_overlap
    return kwargs


def _shifted_iou_benchmark_runs(
    base_pairwise_cost_kwargs: Mapping[str, Any],
) -> list[tuple[dict[str, Any], str]]:
    """Return official shifted-IoU sweep runs and their comparison labels."""

    shifted_specs = (
        (
            "global-registered-shifted-iou-r0",
            "Global registered shifted IoU r=0",
            "registered-shifted-iou",
            "global_registered_shifted_iou_r0.csv",
            0,
            0.0,
            False,
        ),
        (
            "global-registered-shifted-iou-r1",
            "Global registered shifted IoU r=1",
            "registered-shifted-iou",
            "global_registered_shifted_iou_r1.csv",
            1,
            0.0,
            False,
        ),
        (
            "global-registered-shifted-iou-r2",
            "Global registered shifted IoU r=2",
            "registered-shifted-iou",
            "global_registered_shifted_iou_r2.csv",
            2,
            0.0,
            False,
        ),
        (
            "global-registered-shifted-iou-r4",
            "Global registered shifted IoU r=4",
            "registered-shifted-iou",
            "global_registered_shifted_iou_r4.csv",
            4,
            0.0,
            False,
        ),
        (
            "global-registered-shifted-iou-r2-p025",
            "Global registered shifted IoU r=2 p=0.25",
            "registered-shifted-iou",
            "global_registered_shifted_iou_r2_p025.csv",
            2,
            0.25,
            False,
        ),
        (
            "global-roi-aware-shifted-r2-p025",
            "Global ROI-aware shifted r=2 p=0.25",
            "roi-aware-shifted",
            "global_roi_aware_shifted_r2_p025.csv",
            2,
            0.25,
            True,
        ),
    )
    return [
        (
            {
                "name": name,
                "method": "global-assignment",
                "cost": cost,
                "format": "csv",
                "output": output,
                "pairwise_cost_kwargs": _shifted_pairwise_kwargs(
                    base_pairwise_cost_kwargs,
                    radius=radius,
                    shift_penalty_weight=shift_penalty_weight,
                    include_mask_cosine_flag=include_mask_cosine_flag,
                ),
            },
            label,
        )
        for (
            name,
            label,
            cost,
            output,
            radius,
            shift_penalty_weight,
            include_mask_cosine_flag,
        ) in shifted_specs
    ]


def _summary_table(rows: list[dict[str, int | str]]) -> str:
    body = ["| kind | name | rows | output |", "| --- | --- | ---: | --- |"]
    for row in rows:
        body.append(
            f"| {row['kind']} | {row['name']} | {row['rows']} | {row['output']} |"
        )
    return "\n".join(body) + "\n"


def main() -> int:
    results_dir = Path("benchmark-results")
    results_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(os.environ["TRACK2P_DATA_PATH"])
    reference_path = Path(os.environ["TRACK2P_REFERENCE_PATH"])
    if not data_path.exists():
        raise FileNotFoundError(
            f"TRACK2P_DATA_PATH does not exist on this runner: {data_path}"
        )
    if not reference_path.exists():
        raise FileNotFoundError(
            f"TRACK2P_REFERENCE_PATH does not exist on this runner: {reference_path}"
        )

    subject_dirs = discover_subject_dirs(data_path)
    run_calibrated_loso = _should_run_loso(
        os.environ.get("TRACK2P_RUN_CALIBRATED_LOSO", "auto"),
        n_subjects=len(subject_dirs),
    )
    run_shifted_iou_sweep = _bool_env("TRACK2P_RUN_SHIFTED_IOU_SWEEP", default=True)
    base_pairwise_cost_kwargs = _json_object_env("TRACK2P_PAIRWISE_COST_KWARGS_JSON")

    defaults: dict[str, Any] = {
        "data": str(data_path),
        "reference": str(reference_path),
        "reference_kind": os.environ.get("TRACK2P_REFERENCE_KIND", "manual-gt"),
        "allow_track2p_as_reference_for_smoke_test": _bool_env(
            "TRACK2P_ALLOW_SMOKE_REFERENCE"
        ),
        "plane_name": os.environ.get("TRACK2P_PLANE", "plane0"),
        "input_format": os.environ.get("TRACK2P_INPUT_FORMAT", "auto"),
        "include_non_cells": _bool_env("TRACK2P_INCLUDE_NON_CELLS", default=True),
        "include_behavior": _bool_env("TRACK2P_INCLUDE_BEHAVIOR"),
        "max_gap": _int_env("TRACK2P_MAX_GAP", default=2),
        "transform_type": os.environ.get("TRACK2P_TRANSFORM_TYPE", "fov-translation"),
        "seed_session": _int_env("TRACK2P_SEED_SESSION", default=0),
        "restrict_to_reference_seed_rois": _bool_env(
            "TRACK2P_RESTRICT_TO_REFERENCE_SEED_ROIS", default=True
        ),
        "pairwise_cost_kwargs": base_pairwise_cost_kwargs,
    }

    runs: list[dict[str, Any]] = [
        {
            "name": "track2p-baseline",
            "method": "track2p-baseline",
            "format": "csv",
            "output": "track2p_baseline.csv",
        },
        {
            "name": "global-registered-iou",
            "method": "global-assignment",
            "cost": "registered-iou",
            "format": "csv",
            "output": "global_registered_iou.csv",
        },
        {
            "name": "global-roi-aware",
            "method": "global-assignment",
            "cost": "roi-aware",
            "format": "csv",
            "output": "global_roi_aware.csv",
        },
    ]
    comparison_inputs = {
        "Track2p baseline": "track2p-baseline",
        "Global registered IoU": "global-registered-iou",
        "Global ROI-aware": "global-roi-aware",
    }
    shifted_iou_runs: list[tuple[dict[str, Any], str]] = []
    if run_shifted_iou_sweep:
        shifted_iou_runs = _shifted_iou_benchmark_runs(base_pairwise_cost_kwargs)
        for run, label in shifted_iou_runs:
            runs.append(run)
            comparison_inputs[label] = str(run["name"])

    if run_calibrated_loso:
        runs.append(
            {
                "name": "global-calibrated-loso",
                "method": "global-assignment",
                "cost": "calibrated",
                "split": "leave-one-subject-out",
                "format": "csv",
                "output": "global_calibrated_loso.csv",
            }
        )
        comparison_inputs["Global calibrated LOSO"] = "global-calibrated-loso"
        if run_shifted_iou_sweep:
            calibrated_shifted_run = {
                "name": "global-calibrated-loso-shifted-r2-p025",
                "method": "global-assignment",
                "cost": "calibrated",
                "split": "leave-one-subject-out",
                "format": "csv",
                "output": "global_calibrated_loso_shifted_r2_p025.csv",
                "pairwise_cost_kwargs": _shifted_pairwise_kwargs(
                    base_pairwise_cost_kwargs,
                    radius=2,
                    shift_penalty_weight=0.25,
                    include_mask_cosine_flag=True,
                ),
            }
            runs.append(calibrated_shifted_run)
            shifted_iou_runs.append(
                (calibrated_shifted_run, "Global calibrated LOSO shifted r=2 p=0.25")
            )
            comparison_inputs["Global calibrated LOSO shifted r=2 p=0.25"] = str(
                calibrated_shifted_run["name"]
            )

    manifest_data = {
        "defaults": defaults,
        "runs": runs,
        "comparisons": [
            {
                "name": "summary",
                "inputs": comparison_inputs,
                "output": "comparison.md",
                "format": "markdown",
                "highlight_best": True,
            },
            {
                "name": "summary-csv",
                "inputs": comparison_inputs,
                "output": "comparison.csv",
                "format": "csv",
            },
        ],
    }
    metadata = {
        "subjects": [path.name for path in subject_dirs],
        "n_subjects": len(subject_dirs),
        "run_calibrated_loso": run_calibrated_loso,
        "run_shifted_iou_sweep": run_shifted_iou_sweep,
        "shifted_iou_runs": [
            str(run["name"]) for run, _label in shifted_iou_runs
        ],
        "pyrecest_repository": PYRECEST_REPOSITORY,
        "pyrecest_commit": PYRECEST_COMMIT,
    }

    manifest_path = results_dir / "track2p_benchmark_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_data, indent=2) + "\n", encoding="utf-8"
    )
    (results_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    manifest = load_benchmark_manifest(
        manifest_path, output_dir=results_dir, progress=False
    )
    result = run_benchmark_manifest(manifest)
    rows = [
        {"kind": "run", **row} for row in (summary.to_dict() for summary in result.runs)
    ]
    rows.extend(
        {"kind": "comparison", **row}
        for row in (summary.to_dict() for summary in result.comparisons)
    )
    summary = _summary_table(rows)
    (results_dir / "workflow-summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
