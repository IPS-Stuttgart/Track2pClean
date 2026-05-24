"""Runtime bugfix integrations for benchmark and registration QA helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import benchmark_manifest as _benchmark_manifest
from . import registration_qa_report as _registration_qa_report

_ORIGINAL_AUDIT_REFERENCE_LINKS = _registration_qa_report._audit_reference_links


def install_bugfix_integrations() -> None:
    """Install compatibility fixes for existing experiment helpers."""

    _registration_qa_report._linked_source_rois = _linked_source_rois
    _registration_qa_report._gt_affine_oracle_points = _gt_affine_oracle_points
    _registration_qa_report._audit_reference_links = _audit_reference_links
    _benchmark_manifest.run_benchmark_manifest = _run_benchmark_manifest


def _is_present_reference_roi(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return False
    try:
        return int(value) >= 0
    except (TypeError, ValueError, OverflowError):
        return False


def _linked_source_rois(
    reference_matrix: np.ndarray,
    source_index: int,
    target_index: int,
) -> tuple[int, ...]:
    linked_rois: list[int] = []
    seen: set[int] = set()
    for track in reference_matrix:
        source_roi = track[source_index]
        target_roi = track[target_index]
        if not _is_present_reference_roi(source_roi) or not _is_present_reference_roi(
            target_roi
        ):
            continue
        source_roi_int = int(source_roi)
        if source_roi_int in seen:
            continue
        seen.add(source_roi_int)
        linked_rois.append(source_roi_int)
    return tuple(linked_rois)


def _audit_reference_links(
    subject: str,
    source_index: int,
    target_index: int,
    source_session: Any,
    target_session: Any,
    reference_matrix: np.ndarray,
    cost_source_lookup: Any,
    raw_components: Any,
    registered_components: Any,
    cost_matrix: np.ndarray,
    probability_matrix: np.ndarray | None,
    empty_registered_rois: np.ndarray,
    registration_metadata: Any,
    config: Any,
) -> list[dict[str, Any]]:
    sanitized_reference = np.asarray(reference_matrix, dtype=object).copy()
    for track in sanitized_reference:
        if not _is_present_reference_roi(track[source_index]) or not _is_present_reference_roi(
            track[target_index]
        ):
            track[source_index] = None
            track[target_index] = None
    return _ORIGINAL_AUDIT_REFERENCE_LINKS(
        subject,
        source_index,
        target_index,
        source_session,
        target_session,
        sanitized_reference,
        cost_source_lookup,
        raw_components,
        registered_components,
        cost_matrix,
        probability_matrix,
        empty_registered_rois,
        registration_metadata,
        config,
    )


def _gt_affine_oracle_points(
    reference_session: Any,
    target_session: Any,
    reference_matrix: np.ndarray,
    source_index: int,
    target_index: int,
    *,
    weighted_centroids: bool,
) -> tuple[np.ndarray, np.ndarray]:
    source_lookup = _registration_qa_report._roi_lookup(reference_session)
    target_lookup = _registration_qa_report._roi_lookup(target_session)
    source_centroids = reference_session.plane_data.centroids(
        order="yx",
        weighted=weighted_centroids,
    ).T
    target_centroids = target_session.plane_data.centroids(
        order="yx",
        weighted=weighted_centroids,
    ).T
    source_points: list[np.ndarray] = []
    target_points: list[np.ndarray] = []
    for track in reference_matrix:
        source_roi = track[source_index]
        target_roi = track[target_index]
        if not _is_present_reference_roi(source_roi) or not _is_present_reference_roi(
            target_roi
        ):
            continue
        source_roi_int = int(source_roi)
        target_roi_int = int(target_roi)
        if source_roi_int not in source_lookup or target_roi_int not in target_lookup:
            continue
        source_points.append(source_centroids[source_lookup[source_roi_int]])
        target_points.append(target_centroids[target_lookup[target_roi_int]])
    if len(source_points) < 3:
        raise ValueError(
            "transform_type='gt-affine-oracle' requires at least three present "
            "manual-GT links on each audited session edge"
        )
    return np.asarray(source_points, dtype=float), np.asarray(
        target_points, dtype=float
    )


def _run_benchmark_manifest(manifest: Any) -> Any:
    """Run all manifest entries while preserving runner-specific output writers."""

    run_summaries: list[Any] = []
    run_outputs: dict[str, Any] = {}
    for run_spec in manifest.runs:
        rows = _benchmark_manifest._run_benchmark_rows(run_spec)
        _benchmark_manifest._write_run_rows(run_spec, rows)
        run_summaries.append(
            _benchmark_manifest.BenchmarkOutputSummary(
                name=run_spec.name,
                output=run_spec.output,
                rows=len(rows),
            )
        )
        run_outputs[run_spec.name] = run_spec.output

    comparison_summaries: list[Any] = []
    for comparison_spec in manifest.comparisons:
        comparison_inputs = _benchmark_manifest._comparison_inputs(
            comparison_spec, run_outputs=run_outputs, manifest_path=manifest.path
        )
        rows = _benchmark_manifest.aggregate_rows(
            _benchmark_manifest.load_labeled_rows(comparison_inputs)
        )
        _benchmark_manifest.write_comparison(
            rows,
            comparison_spec.output,
            comparison_spec.output_format,
            highlight_best=comparison_spec.highlight_best,
        )
        comparison_summaries.append(
            _benchmark_manifest.BenchmarkOutputSummary(
                name=comparison_spec.name,
                output=comparison_spec.output,
                rows=len(rows),
            )
        )

    return _benchmark_manifest.BenchmarkManifestResult(
        runs=tuple(run_summaries), comparisons=tuple(comparison_summaries)
    )


__all__ = ["install_bugfix_integrations"]
