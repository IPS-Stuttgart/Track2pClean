"""Generate and run the guarded Track2p benchmark suite for GitHub Actions."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bayescatrack.dependency_pins import PYRECEST_COMMIT, PYRECEST_REPOSITORY
from bayescatrack.experiments.benchmark_manifest import (
    load_benchmark_manifest,
    run_benchmark_manifest,
)
from bayescatrack.experiments.track2p_benchmark import discover_subject_dirs


@dataclass(frozen=True)
class BenchmarkGateResult:
    """One optional regression gate evaluated from the comparison CSV."""

    metric: str
    condition: str
    approach: str
    observed: float
    threshold: float
    passed: bool


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


def _optional_float_env(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a floating-point threshold") from exc


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


def _summary_table(rows: list[dict[str, int | str]]) -> str:
    body = [
        "## Track2p benchmark artifacts",
        "",
        "| kind | name | rows | output |",
        "| --- | --- | ---: | --- |",
    ]
    for row in rows:
        body.append(
            f"| {row['kind']} | {row['name']} | {row['rows']} | {row['output']} |"
        )
    return "\n".join(body) + "\n"


def _load_comparison_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _float_cell(row: dict[str, str], column: str) -> float:
    try:
        return float(row[column])
    except KeyError as exc:
        raise KeyError(f"Comparison CSV is missing required column {column!r}") from exc
    except ValueError as exc:
        approach = row.get("approach", "<unknown>")
        raise ValueError(
            f"Comparison CSV value for {column!r} in approach {approach!r} is not numeric"
        ) from exc


def _find_row(rows: list[dict[str, str]], approach: str) -> dict[str, str]:
    for row in rows:
        if row.get("approach") == approach:
            return row
    available = ", ".join(row.get("approach", "<unknown>") for row in rows)
    raise ValueError(
        f"Reference approach {approach!r} was not found in comparison.csv. "
        f"Available approaches: {available}"
    )


def _best_non_reference_row(
    rows: list[dict[str, str]], *, metric_column: str, reference_approach: str
) -> dict[str, str]:
    candidates = [row for row in rows if row.get("approach") != reference_approach]
    if not candidates:
        raise ValueError("At least one non-reference benchmark approach is required")
    return max(candidates, key=lambda row: _float_cell(row, metric_column))


def _gate_result(
    *,
    rows: list[dict[str, str]],
    metric_column: str,
    threshold: float,
    condition: str,
    reference_approach: str,
    delta_over_reference: bool,
) -> BenchmarkGateResult:
    best = _best_non_reference_row(
        rows, metric_column=metric_column, reference_approach=reference_approach
    )
    best_value = _float_cell(best, metric_column)
    observed = best_value
    if delta_over_reference:
        reference = _find_row(rows, reference_approach)
        observed = best_value - _float_cell(reference, metric_column)
    return BenchmarkGateResult(
        metric=metric_column,
        condition=condition,
        approach=str(best.get("approach", "<unknown>")),
        observed=observed,
        threshold=threshold,
        passed=observed >= threshold,
    )


def _evaluate_regression_gates(comparison_csv: Path) -> list[BenchmarkGateResult]:
    rows = _load_comparison_rows(comparison_csv)
    configured_thresholds = {
        "TRACK2P_MIN_BEST_PAIRWISE_F1_MACRO": _optional_float_env(
            "TRACK2P_MIN_BEST_PAIRWISE_F1_MACRO"
        ),
        "TRACK2P_MIN_BEST_COMPLETE_TRACK_F1_MACRO": _optional_float_env(
            "TRACK2P_MIN_BEST_COMPLETE_TRACK_F1_MACRO"
        ),
        "TRACK2P_MIN_PAIRWISE_F1_MACRO_DELTA_OVER_BASELINE": _optional_float_env(
            "TRACK2P_MIN_PAIRWISE_F1_MACRO_DELTA_OVER_BASELINE"
        ),
        "TRACK2P_MIN_COMPLETE_TRACK_F1_MACRO_DELTA_OVER_BASELINE": _optional_float_env(
            "TRACK2P_MIN_COMPLETE_TRACK_F1_MACRO_DELTA_OVER_BASELINE"
        ),
    }
    if all(threshold is None for threshold in configured_thresholds.values()):
        return []
    if not rows:
        raise ValueError(
            f"Regression gates were requested, but no comparison CSV exists at {comparison_csv}"
        )

    reference_approach = os.environ.get(
        "TRACK2P_BASELINE_APPROACH", "Track2p baseline"
    ).strip()
    if not reference_approach:
        raise ValueError("TRACK2P_BASELINE_APPROACH must not be empty")

    gate_specs = (
        (
            configured_thresholds["TRACK2P_MIN_BEST_PAIRWISE_F1_MACRO"],
            "pairwise_f1_macro",
            "best non-baseline macro pairwise F1 >= threshold",
            False,
        ),
        (
            configured_thresholds["TRACK2P_MIN_BEST_COMPLETE_TRACK_F1_MACRO"],
            "complete_track_f1_macro",
            "best non-baseline macro complete-track F1 >= threshold",
            False,
        ),
        (
            configured_thresholds["TRACK2P_MIN_PAIRWISE_F1_MACRO_DELTA_OVER_BASELINE"],
            "pairwise_f1_macro",
            "best non-baseline macro pairwise F1 delta over baseline >= threshold",
            True,
        ),
        (
            configured_thresholds[
                "TRACK2P_MIN_COMPLETE_TRACK_F1_MACRO_DELTA_OVER_BASELINE"
            ],
            "complete_track_f1_macro",
            "best non-baseline macro complete-track F1 delta over baseline >= threshold",
            True,
        ),
    )
    return [
        _gate_result(
            rows=rows,
            metric_column=metric_column,
            threshold=float(threshold),
            condition=condition,
            reference_approach=reference_approach,
            delta_over_reference=delta_over_reference,
        )
        for threshold, metric_column, condition, delta_over_reference in gate_specs
        if threshold is not None
    ]


def _format_gate_summary(gates: list[BenchmarkGateResult]) -> str:
    if not gates:
        return "## Regression gates\n\nNo regression thresholds were configured.\n"
    body = [
        "## Regression gates",
        "",
        "| status | metric | condition | approach | observed | threshold |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for gate in gates:
        status = "PASS" if gate.passed else "FAIL"
        body.append(
            f"| {status} | {gate.metric} | {gate.condition} | {gate.approach} | "
            f"{gate.observed:.6f} | {gate.threshold:.6f} |"
        )
    return "\n".join(body) + "\n"


def _append_file_section(sections: list[str], *, title: str, path: Path) -> None:
    if not path.exists():
        return
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return
    sections.append(f"## {title}\n\n{content}\n")


def _write_workflow_summary(
    *,
    results_dir: Path,
    artifact_rows: list[dict[str, int | str]],
    gates: list[BenchmarkGateResult],
) -> str:
    sections = [
        _summary_table(artifact_rows).strip(),
        _format_gate_summary(gates).strip(),
    ]
    _append_file_section(
        sections, title="Comparison", path=results_dir / "comparison.md"
    )
    summary = "\n\n".join(sections).strip() + "\n"
    (results_dir / "workflow-summary.md").write_text(summary, encoding="utf-8")
    return summary


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
    include_policy_dp_experiment = _bool_env(
        "TRACK2P_INCLUDE_POLICY_DP_EXPERIMENT", default=False
    )
    include_policy_pruned_experiment = _bool_env(
        "TRACK2P_INCLUDE_POLICY_PRUNED_EXPERIMENT", default=False
    )

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
        "pairwise_cost_kwargs": _json_object_env("TRACK2P_PAIRWISE_COST_KWARGS_JSON"),
    }

    runs: list[dict[str, Any]] = [
        {
            "name": "track2p-baseline",
            "method": "track2p-baseline",
            "format": "csv",
            "output": "track2p_baseline.csv",
        },
        {
            "name": "track2p-policy",
            "runner": "track2p-policy",
            "format": "csv",
            "transform_type": "affine",
            "threshold_method": "min",
            "iou_distance_threshold": 12.0,
            "cell_probability_threshold": 0.5,
            "max_gap": 1,
            "output": "track2p_policy.csv",
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
        "Track2p policy": "track2p-policy",
        "Global registered IoU": "global-registered-iou",
        "Global ROI-aware": "global-roi-aware",
    }
    if include_policy_dp_experiment:
        runs.append(
            {
                "name": "track2p-policy-dp",
                "runner": "track2p-policy-dp",
                "format": "csv",
                "transform_type": "affine",
                "threshold_method": "min",
                "iou_distance_threshold": 12.0,
                "cell_probability_threshold": 0.5,
                "row_top_k": 2,
                "rescue_min_iou": 0.10,
                "threshold_rescue_margin": 0.15,
                "beam_width": 8,
                "max_gap": 2,
                "output": "track2p_policy_dp_experimental.csv",
            }
        )
        comparison_inputs.update({"Track2p policy DP": "track2p-policy-dp"})
    if include_policy_pruned_experiment:
        runs.append(
            {
                "name": "track2p-policy-pruned",
                "runner": "track2p-policy-pruned",
                "format": "csv",
                "output": "track2p_policy_pruned.csv",
                "threshold_method": "min",
                "iou_distance_threshold": 12.0,
                "prune_threshold_margin": 0.02,
                "prune_competition_margin": 0.02,
                "prune_min_area_ratio": 0.45,
                "prune_centroid_distance": 10.0,
            }
        )
        comparison_inputs["Track2p policy pruned"] = "track2p-policy-pruned"
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
        "include_policy_dp_experiment": include_policy_dp_experiment,
        "include_policy_pruned_experiment": include_policy_pruned_experiment,
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
    artifact_rows = [
        {"kind": "run", **row} for row in (summary.to_dict() for summary in result.runs)
    ]
    artifact_rows.extend(
        {"kind": "comparison", **row}
        for row in (summary.to_dict() for summary in result.comparisons)
    )
    gates = _evaluate_regression_gates(results_dir / "comparison.csv")
    summary = _write_workflow_summary(
        results_dir=results_dir,
        artifact_rows=artifact_rows,
        gates=gates,
    )
    print(summary)
    failed_gates = [gate for gate in gates if not gate.passed]
    if failed_gates:
        raise SystemExit(
            f"{len(failed_gates)} Track2p benchmark regression gate(s) failed"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
