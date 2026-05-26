from __future__ import annotations

import argparse
import importlib
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from bayescatrack import CalciumPlaneData, load_suite2p_plane
from bayescatrack.association.dynamic_edge_priors import (
    DynamicEdgePriorConfig,
    apply_dynamic_edge_priors,
)
from bayescatrack.fov_affine_registration import (
    apply_affine_image_warp,
    apply_affine_roi_mask_warp,
    register_measurement_plane_by_fov_affine,
)
from bayescatrack.fov_registration import apply_integer_image_translation
from bayescatrack.matching import solve_bundle_linear_assignment
from bayescatrack.reference import score_complete_tracks, score_pairwise_matches
from bayescatrack.registration import RegisteredConsecutiveBundles
from bayescatrack.tracking import SubjectTrackingResult


def _spot_image(
    shape: tuple[int, int], centers: tuple[tuple[int, int], ...]
) -> np.ndarray:
    image = np.zeros(shape, dtype=float)
    for y, x in centers:
        image[y - 1 : y + 2, x - 1 : x + 2] = 1.0
    return image


def test_importing_package_does_not_mutate_argparse_add_argument():
    original = argparse.ArgumentParser.add_argument
    import bayescatrack

    importlib.reload(bayescatrack)

    assert argparse.ArgumentParser.add_argument is original


def _pairwise_cost_patch_count(method, marker: str) -> int:
    count = 0
    seen: set[int] = set()
    current = method
    while current is not None:
        current_id = id(current)
        if current_id in seen:
            raise AssertionError("cycle in pairwise cost wrapper chain")
        seen.add(current_id)
        if getattr(current, marker, False):
            count += 1
        current = getattr(current, "_bayescatrack_original", None)
    return count


def test_pairwise_cost_patch_installers_are_reload_idempotent():
    import bayescatrack

    importlib.reload(bayescatrack)
    importlib.reload(bayescatrack)

    method = bayescatrack.CalciumPlaneData.build_pairwise_cost_matrix
    assert _pairwise_cost_patch_count(method, "_bayescatrack_soft_overlap_patch") == 1
    assert _pairwise_cost_patch_count(method, "_bayescatrack_advanced_roi_patch") == 1


def test_fov_affine_registration_keeps_registered_measurement_fov():
    reference_fov = _spot_image((96, 96), ((20, 22), (28, 72), (68, 28), (72, 75)))
    measurement_fov = apply_integer_image_translation(
        reference_fov, [-3, 4], output_shape=(96, 96)
    )
    reference_plane = CalciumPlaneData(
        reference_fov[None, :, :] > 0.0, fov=reference_fov, source="reference"
    )
    measurement_plane = CalciumPlaneData(
        measurement_fov[None, :, :] > 0.0, fov=measurement_fov, source="measurement"
    )

    registration = register_measurement_plane_by_fov_affine(
        reference_plane, measurement_plane
    )

    assert registration.registered_measurement_plane.fov is not reference_plane.fov
    assert registration.registered_measurement_plane.fov.shape == reference_fov.shape


def test_affine_image_and_mask_warps_use_inverse_resampling():
    image = np.zeros((8, 8), dtype=float)
    image[2:5, 3:6] = 1.0
    mask = image[None, :, :] > 0.0
    matrix_xy = np.asarray([[1.0, 0.25, -1.0], [0.15, 1.0, 0.5]], dtype=float)

    warped_image = apply_affine_image_warp(image, matrix_xy, output_shape=(8, 8))
    warped_mask = apply_affine_roi_mask_warp(mask, matrix_xy, output_shape=(8, 8))

    assert warped_image.shape == image.shape
    assert warped_mask.shape == mask.shape
    assert warped_mask.dtype == mask.dtype
    assert np.count_nonzero(warped_mask) > 0


def test_linear_assignment_ignores_nonfinite_costs_when_ungated():
    bundle = SimpleNamespace(
        pairwise_cost_matrix=np.asarray([[np.nan, 1.0], [2.0, np.inf]]),
        reference_roi_indices=np.asarray([10, 11]),
        measurement_roi_indices=np.asarray([20, 21]),
        reference_session_name="day0",
        measurement_session_name="day1",
    )

    result = solve_bundle_linear_assignment(bundle, max_cost=None)

    np.testing.assert_array_equal(
        result.as_pair_array(), np.asarray([[10, 21], [11, 20]])
    )
    np.testing.assert_allclose(result.costs, np.asarray([1.0, 2.0]))


def test_dynamic_edge_priors_accept_compacted_registered_roi_columns():
    full_empty_registered_rois = np.asarray([False, True, False])
    config = DynamicEdgePriorConfig(registration_empty_roi_weight=8.0)

    full_layout_costs = np.asarray([[1.0, 2.0, 3.0]], dtype=float)
    adjusted_full_layout = apply_dynamic_edge_priors(
        full_layout_costs,
        {},
        session_gap=1,
        empty_registered_rois=full_empty_registered_rois,
        config=config,
    )
    np.testing.assert_allclose(
        adjusted_full_layout,
        np.asarray([[1.0, 10.0, 3.0]], dtype=float),
    )

    compact_layout_costs = np.asarray([[1.0, 3.0]], dtype=float)
    adjusted_compact_layout = apply_dynamic_edge_priors(
        compact_layout_costs,
        {},
        session_gap=1,
        empty_registered_rois=full_empty_registered_rois,
        config=config,
    )

    np.testing.assert_allclose(adjusted_compact_layout, compact_layout_costs)


def test_tracking_coverage_ratios_are_nan_when_denominator_is_zero():
    result = SubjectTrackingResult(
        sessions=(),
        registered_bundles=RegisteredConsecutiveBundles(bundles=[]),
        match_results=(),
        session_names=("day0", "day1"),
        track_rows=np.zeros((0, 2), dtype=int),
        link_costs=np.zeros((0, 1), dtype=float),
    )

    assert np.isnan(result.score_summary()["complete_track_fraction"])


def test_reference_scores_are_nan_when_no_metric_denominator_exists():
    pair_scores = score_pairwise_matches(
        np.zeros((0, 2), dtype=int), np.zeros((0, 2), dtype=int)
    )
    complete_scores = score_complete_tracks(
        np.zeros((0, 2), dtype=int), np.zeros((0, 2), dtype=int)
    )

    assert np.isnan(pair_scores["precision"])
    assert np.isnan(pair_scores["recall"])
    assert np.isnan(pair_scores["f1"])
    assert complete_scores["ct"] == 1.0


def test_load_suite2p_plane_infers_shape_after_empty_stat_entries(tmp_path):
    stat = np.asarray(
        [
            {
                "ypix": np.asarray([], dtype=int),
                "xpix": np.asarray([], dtype=int),
                "lam": np.asarray([], dtype=float),
            },
            {
                "ypix": np.asarray([2], dtype=int),
                "xpix": np.asarray([3], dtype=int),
                "lam": np.asarray([1.0], dtype=float),
            },
        ],
        dtype=object,
    )
    np.save(tmp_path / "stat.npy", stat)

    plane = load_suite2p_plane(
        tmp_path,
        load_traces=False,
        load_spike_traces=False,
    )

    assert plane.roi_masks.shape == (1, 3, 4)
    assert plane.roi_indices.tolist() == [1]


def test_load_suite2p_plane_rejects_all_empty_stats_without_ops(tmp_path):
    stat = np.asarray(
        [
            {
                "ypix": np.asarray([], dtype=int),
                "xpix": np.asarray([], dtype=int),
                "lam": np.asarray([], dtype=float),
            }
        ],
        dtype=object,
    )
    np.save(tmp_path / "stat.npy", stat)

    with pytest.raises(ValueError, match="all ROI pixel arrays are empty"):
        load_suite2p_plane(
            tmp_path,
            load_traces=False,
            load_spike_traces=False,
        )


def test_load_suite2p_plane_rejects_mismatched_iscell_rows(tmp_path):
    stat = np.asarray(
        [
            {
                "ypix": np.asarray([0], dtype=int),
                "xpix": np.asarray([0], dtype=int),
                "lam": np.asarray([1.0], dtype=float),
            },
            {
                "ypix": np.asarray([1], dtype=int),
                "xpix": np.asarray([1], dtype=int),
                "lam": np.asarray([1.0], dtype=float),
            },
        ],
        dtype=object,
    )
    np.save(tmp_path / "stat.npy", stat)
    np.save(tmp_path / "iscell.npy", np.ones((1, 2), dtype=float))

    with pytest.raises(ValueError, match="iscell.npy first dimension"):
        load_suite2p_plane(tmp_path, load_traces=False, load_spike_traces=False)


def test_load_suite2p_plane_accepts_single_column_iscell_without_scalar_warning(
    tmp_path,
):
    stat = np.asarray(
        [
            {
                "ypix": np.asarray([0], dtype=int),
                "xpix": np.asarray([0], dtype=int),
                "lam": np.asarray([1.0], dtype=float),
            },
            {
                "ypix": np.asarray([1], dtype=int),
                "xpix": np.asarray([1], dtype=int),
                "lam": np.asarray([1.0], dtype=float),
            },
        ],
        dtype=object,
    )
    np.save(tmp_path / "stat.npy", stat)
    np.save(tmp_path / "iscell.npy", np.asarray([[1.0], [0.0]], dtype=float))

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        plane = load_suite2p_plane(
            tmp_path,
            load_traces=False,
            load_spike_traces=False,
        )

    assert plane.roi_indices.tolist() == [0]
    np.testing.assert_allclose(plane.cell_probabilities, np.asarray([1.0]))


def test_load_suite2p_plane_rejects_mismatched_trace_rows(tmp_path):
    stat = np.asarray(
        [
            {
                "ypix": np.asarray([0], dtype=int),
                "xpix": np.asarray([0], dtype=int),
                "lam": np.asarray([1.0], dtype=float),
            },
            {
                "ypix": np.asarray([1], dtype=int),
                "xpix": np.asarray([1], dtype=int),
                "lam": np.asarray([1.0], dtype=float),
            },
        ],
        dtype=object,
    )
    np.save(tmp_path / "stat.npy", stat)
    np.save(tmp_path / "F.npy", np.ones((1, 3), dtype=float))

    with pytest.raises(ValueError, match="F.npy first dimension"):
        load_suite2p_plane(tmp_path, load_spike_traces=False)


def test_pairwise_feature_dimension_mismatch_raises_clear_error():
    masks = np.zeros((1, 4, 4), dtype=bool)
    masks[0, 1:3, 1:3] = True
    reference = CalciumPlaneData(
        masks,
        roi_features={"embedding": np.zeros((1, 2), dtype=float)},
    )
    measurement = CalciumPlaneData(
        masks,
        roi_features={"embedding": np.zeros((1, 1), dtype=float)},
    )

    with pytest.raises(ValueError, match="incompatible trailing dimensions"):
        reference.build_pairwise_cost_matrix(
            measurement,
            centroid_weight=0.0,
            iou_weight=0.0,
            mask_cosine_weight=0.0,
            area_weight=0.0,
            roi_feature_weight=1.0,
            feature_names=("embedding",),
        )


def test_return_components_records_advanced_roi_components():
    masks = np.zeros((1, 4, 4), dtype=bool)
    masks[0, 1:3, 1:3] = True
    plane = CalciumPlaneData(masks)

    _, components = plane.build_pairwise_cost_matrix(plane, return_components=True)

    assert "radial_profile_cost" in components
    assert "orientation_cost" in components
    assert "ambiguity_margin_cost" in components
