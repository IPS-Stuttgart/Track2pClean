import json

import numpy as np
import numpy.testing as npt
import pytest
from bayescatrack import CalciumPlaneData, load_suite2p_plane, load_track2p_subject
from bayescatrack.association.calibrated_costs import (
    DEFAULT_ASSOCIATION_FEATURES,
    LOCAL_EVIDENCE_ASSOCIATION_FEATURES,
    pairwise_feature_tensor,
)
from bayescatrack.experiments.track2p_loso_calibration import (
    calibration_feature_names,
    pairwise_cost_kwargs_for_calibration_features,
)
from tests._support import run_module


def _write_minimal_raw_npy_plane(plane_dir):
    plane_dir.mkdir(parents=True)
    roi_masks: np.ndarray = np.zeros((1, 2, 2), dtype=bool)
    roi_masks[0, 0, 1] = True
    np.save(plane_dir / "rois.npy", roi_masks)
    np.save(plane_dir / "F.npy", np.array([[1.0, 2.0, 3.0]], dtype=float))
    np.save(plane_dir / "fov.npy", np.ones((2, 2), dtype=float))


def test_calcium_plane_data_builds_measurements_and_state_moments():
    roi_masks: np.ndarray = np.zeros((1, 3, 3), dtype=bool)
    roi_masks[0, 1, 2] = True
    plane = CalciumPlaneData(roi_masks=roi_masks)

    npt.assert_allclose(
        plane.to_measurement_matrix(order="xy"), np.array([[2.0], [1.0]])
    )
    means, covariances = plane.to_constant_velocity_state_moments(order="xy")

    assert means.shape == (4, 1)
    assert covariances.shape == (4, 4, 1)


def test_mask_geometry_ignores_non_finite_and_negative_weights():
    roi_masks: np.ndarray = np.zeros((1, 3, 3), dtype=float)
    roi_masks[0, 0, 0] = 1.0
    roi_masks[0, 0, 2] = np.nan
    roi_masks[0, 2, 0] = -5.0
    plane = CalciumPlaneData(roi_masks=roi_masks)

    npt.assert_allclose(plane.roi_areas(weighted=False), np.array([1.0]))
    npt.assert_allclose(plane.roi_areas(weighted=True), np.array([1.0]))
    npt.assert_allclose(
        plane.centroids(order="xy", weighted=False),
        np.array([[0.0], [0.0]]),
    )
    npt.assert_allclose(
        plane.centroids(order="xy", weighted=True),
        np.array([[0.0], [0.0]]),
    )


def test_roi_feature_cost_scales_multi_dimensional_features_per_dimension():
    reference_masks = np.zeros((1, 5, 5), dtype=bool)
    reference_masks[0, 1:3, 1:3] = True
    measurement_masks = np.repeat(reference_masks, 2, axis=0)
    reference = CalciumPlaneData(
        roi_masks=reference_masks,
        roi_features={
            "shape_embedding": np.array(
                [
                    [0.0, 0.0],
                ],
                dtype=float,
            )
        },
    )
    measurement = CalciumPlaneData(
        roi_masks=measurement_masks,
        roi_features={
            "shape_embedding": np.array(
                [
                    [0.2, 0.0],
                    [0.0, 2000.0],
                ],
                dtype=float,
            )
        },
    )

    _, components = reference.build_pairwise_cost_matrix(
        measurement,
        centroid_weight=0.0,
        iou_weight=0.0,
        mask_cosine_weight=0.0,
        area_weight=0.0,
        roi_feature_weight=1.0,
        feature_names=("shape_embedding",),
        cell_probability_weight=0.0,
        return_components=True,
    )

    feature_cost = components["roi_feature_cost"]
    assert feature_cost.shape == (1, 2)
    assert np.all(np.isfinite(feature_cost))
    assert feature_cost[0, 0] > 0.25 * feature_cost[0, 1]
    assert feature_cost[0, 1] > 0.25 * feature_cost[0, 0]


def test_local_image_and_weighted_mask_evidence_components_rank_diagonal_pairs():
    image_shape = (12, 12)
    reference_masks: np.ndarray = np.zeros((2, *image_shape), dtype=float)
    measurement_masks: np.ndarray = np.zeros((2, *image_shape), dtype=float)
    reference_masks[0, 2:4, 2:4] = np.array([[1.0, 2.0], [3.0, 4.0]])
    measurement_masks[0, 2:4, 2:4] = np.array([[4.0, 3.0], [2.0, 1.0]])
    reference_masks[1, 7:9, 7:9] = 1.0
    measurement_masks[1, 7:9, 7:9] = 1.0

    fov: np.ndarray = np.zeros(image_shape, dtype=float)
    fov[3, 1:6] = 1.0
    fov[6:11, 8] = 1.0
    reference = CalciumPlaneData(roi_masks=reference_masks, fov=fov)
    measurement = CalciumPlaneData(roi_masks=measurement_masks, fov=fov.copy())

    # Local-evidence kwargs are installed by BayesCaTrack's core bridge support.
    # pylint: disable=unexpected-keyword-arg
    cost, components = reference.build_pairwise_cost_matrix(
        measurement,
        centroid_weight=0.0,
        iou_weight=0.0,
        mask_cosine_weight=0.0,
        weighted_dice_weight=1.0,
        overlap_fraction_weight=1.0,
        distance_transform_weight=1.0,
        image_patch_weight=1.0,
        centroid_rank_weight=1.0,
        area_weight=0.0,
        roi_feature_weight=0.0,
        cell_probability_weight=0.0,
        local_evidence_components=True,
        patch_radius=2,
        return_components=True,
    )

    assert cost.shape == (2, 2)
    assert np.all(np.isfinite(cost))
    assert (
        components["weighted_dice_similarity"][0, 0]
        > components["weighted_dice_similarity"][0, 1]
    )
    assert (
        components["overlap_min_fraction"][0, 0]
        > components["overlap_min_fraction"][0, 1]
    )
    assert (
        components["distance_transform_cost"][0, 0]
        < components["distance_transform_cost"][0, 1]
    )
    assert components["image_patch_cost"][0, 0] < components["image_patch_cost"][0, 1]
    assert components["centroid_rank_cost"][0, 0] == 0.0
    assert cost[0, 0] < cost[0, 1]
    assert cost[1, 1] < cost[1, 0]

    features = pairwise_feature_tensor(
        components,
        feature_names=LOCAL_EVIDENCE_ASSOCIATION_FEATURES,
    )
    assert features.shape == (2, 2, len(LOCAL_EVIDENCE_ASSOCIATION_FEATURES))
    assert features[0, 0, 0] < features[0, 1, 0]


def test_image_patch_weight_is_neutral_without_fov():
    roi_masks: np.ndarray = np.zeros((1, 5, 5), dtype=bool)
    roi_masks[0, 2:4, 2:4] = True
    reference = CalciumPlaneData(roi_masks=roi_masks)
    measurement = CalciumPlaneData(roi_masks=roi_masks.copy())
    base_kwargs = {
        "centroid_weight": 0.0,
        "iou_weight": 0.0,
        "mask_cosine_weight": 0.0,
        "area_weight": 0.0,
        "roi_feature_weight": 0.0,
        "cell_probability_weight": 0.0,
    }

    base_cost = reference.build_pairwise_cost_matrix(measurement, **base_kwargs)
    patch_cost, components = reference.build_pairwise_cost_matrix(
        measurement,
        image_patch_weight=1.0,
        local_evidence_components=True,
        return_components=True,
        **base_kwargs,
    )

    npt.assert_allclose(patch_cost, base_cost)
    npt.assert_allclose(components["image_patch_cost"], np.zeros((1, 1)))
    npt.assert_allclose(components["image_patch_valid"], np.zeros((1, 1)))


def test_local_evidence_calibration_feature_preset_enables_components():
    default_features = calibration_feature_names("default")
    local_features = calibration_feature_names("local-evidence")
    combined_features = calibration_feature_names("default+local-evidence")

    assert default_features == tuple(DEFAULT_ASSOCIATION_FEATURES)
    assert local_features == tuple(LOCAL_EVIDENCE_ASSOCIATION_FEATURES)
    assert combined_features[: len(DEFAULT_ASSOCIATION_FEATURES)] == tuple(
        DEFAULT_ASSOCIATION_FEATURES
    )
    assert set(LOCAL_EVIDENCE_ASSOCIATION_FEATURES).issubset(combined_features)

    default_kwargs = pairwise_cost_kwargs_for_calibration_features({}, default_features)
    local_kwargs = pairwise_cost_kwargs_for_calibration_features(
        {"patch_radius": 3}, combined_features
    )

    assert default_kwargs is None
    assert local_kwargs == {"patch_radius": 3, "local_evidence_components": True}


def test_load_track2p_subject_auto_warns_before_skipping_missing_session(tmp_path):
    subject_dir = tmp_path / "jm123"
    _write_minimal_raw_npy_plane(subject_dir / "2024-05-01_a" / "data_npy" / "plane0")
    (subject_dir / "2024-05-02_missing").mkdir(parents=True)

    with pytest.warns(RuntimeWarning, match="2024-05-02_missing"):
        sessions = load_track2p_subject(subject_dir)

    assert [session.session_name for session in sessions] == ["2024-05-01_a"]


def test_load_track2p_subject_auto_strict_raises_on_missing_session(tmp_path):
    subject_dir = tmp_path / "jm123"
    (subject_dir / "2024-05-02_missing").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="2024-05-02_missing"):
        load_track2p_subject(subject_dir, strict=True)


def test_cli_summary_and_export(tmp_path):
    subject_dir = tmp_path / "jm123"
    plane_dir = subject_dir / "2024-05-01_a" / "data_npy" / "plane0"
    plane_dir.mkdir(parents=True)
    roi_masks: np.ndarray = np.zeros((1, 2, 2), dtype=bool)
    roi_masks[0, 0, 1] = True
    np.save(plane_dir / "rois.npy", roi_masks)
    np.save(plane_dir / "F.npy", np.array([[1.0, 2.0, 3.0]], dtype=float))
    np.save(plane_dir / "fov.npy", np.ones((2, 2), dtype=float))

    summary_proc = run_module("-m", "bayescatrack", "summary", str(subject_dir))
    summary = json.loads(summary_proc.stdout)
    assert summary["n_sessions"] == 1
    assert summary["sessions"][0]["n_rois"] == 1

    output_path = tmp_path / "jm123_plane0.npz"
    export_proc = run_module(
        "-m", "bayescatrack", "export", str(subject_dir), str(output_path)
    )
    export_summary = json.loads(export_proc.stdout)
    assert export_summary["n_sessions"] == 1
    assert output_path.exists()

    with np.load(output_path, allow_pickle=False) as exported:
        for metadata_key in (
            "session_names",
            "session_dates",
            "plane_name",
            "input_format",
        ):
            assert exported[metadata_key].dtype.kind == "U"

        assert exported["session_names"].tolist() == ["2024-05-01_a"]
        assert exported["session_dates"].tolist() == ["2024-05-01"]


def test_load_suite2p_plane_validates_auxiliary_roi_row_counts(tmp_path):
    plane_dir = tmp_path / "plane0"
    plane_dir.mkdir()

    stat = np.empty(2, dtype=object)
    stat[0] = {"ypix": np.array([0]), "xpix": np.array([0]), "lam": np.array([1.0])}
    stat[1] = {"ypix": np.array([1]), "xpix": np.array([1]), "lam": np.array([1.0])}
    np.save(plane_dir / "stat.npy", stat)
    np.save(plane_dir / "ops.npy", {"Ly": 2, "Lx": 2, "meanImg": np.ones((2, 2))})

    np.save(plane_dir / "iscell.npy", np.array([[1.0, 0.9]], dtype=float))
    with pytest.raises(ValueError, match="iscell.npy.*same number of ROIs"):
        load_suite2p_plane(plane_dir)

    np.save(plane_dir / "iscell.npy", np.array([[1.0, 0.9], [1.0, 0.8]], dtype=float))
    np.save(plane_dir / "F.npy", np.ones((1, 3), dtype=float))
    with pytest.raises(ValueError, match="F.npy.*same number of ROIs"):
        load_suite2p_plane(plane_dir)

    np.save(plane_dir / "F.npy", np.ones((2, 3), dtype=float))
    np.save(plane_dir / "spks.npy", np.ones((1, 3), dtype=float))
    with pytest.raises(ValueError, match="spks.npy.*same number of ROIs"):
        load_suite2p_plane(plane_dir)

    np.save(plane_dir / "spks.npy", np.ones((2, 3), dtype=float))
    assert load_suite2p_plane(plane_dir).n_rois == 2
