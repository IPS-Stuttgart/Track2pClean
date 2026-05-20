from __future__ import annotations

import argparse
import importlib
from types import SimpleNamespace

import numpy as np

from bayescatrack import CalciumPlaneData
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


def _spot_image(shape: tuple[int, int], centers: tuple[tuple[int, int], ...]) -> np.ndarray:
    image = np.zeros(shape, dtype=float)
    for y, x in centers:
        image[y - 1 : y + 2, x - 1 : x + 2] = 1.0
    return image


def test_importing_package_does_not_mutate_argparse_add_argument():
    original = argparse.ArgumentParser.add_argument
    import bayescatrack

    importlib.reload(bayescatrack)

    assert argparse.ArgumentParser.add_argument is original


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
    assert np.isnan(complete_scores["ct"])
