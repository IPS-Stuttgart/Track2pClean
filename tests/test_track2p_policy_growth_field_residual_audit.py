from __future__ import annotations

import numpy as np
import pytest
from bayescatrack import cli
from bayescatrack.experiments import (
    track2p_policy_growth_field_residual_audit as growth_audit,
)


def test_growth_field_residual_audit_is_registered() -> None:
    canonical = cli._BENCHMARK_ALIASES["track2p-growth-field-residual-audit"]

    assert canonical == "track2p-policy-growth-field-residual-audit"
    assert (
        cli._BENCHMARK_ALIASES["track2p-component-growth-field-residual-audit"]
        == canonical
    )
    assert (
        cli._BENCHMARK_COMMANDS[canonical].module
        == "bayescatrack.experiments.track2p_policy_growth_field_residual_audit"
    )


def test_growth_field_residual_audit_parser_exposes_anchor_defaults() -> None:
    args = growth_audit.build_arg_parser().parse_args(
        [
            "--data",
            "track2p-root",
            "--output",
            "growth_edges.csv",
            "--summary-output",
            "growth_summary.csv",
        ]
    )

    assert args.threshold_method == "min"
    assert args.iou_distance_threshold == 12.0
    assert args.anchor_min_registered_iou == 0.50
    assert args.anchor_min_shifted_iou == 0.30
    assert args.anchor_min_cell_probability == 0.80


@pytest.mark.parametrize(
    "option",
    ["--suffix-path-length", "--max-stitches-per-subject", "--edge-top-k", "--path-beam-width"],
)
def test_growth_field_residual_audit_parser_rejects_nonpositive_search_budgets(
    option: str,
) -> None:
    with pytest.raises(SystemExit):
        growth_audit.build_arg_parser().parse_args(
            [
                "--data",
                "track2p-root",
                "--reference",
                "manual-gt",
                "--output",
                "growth_edges.csv",
                option,
                "0",
            ]
        )


def test_fit_growth_model_recovers_affine_translation(monkeypatch) -> None:
    source = {
        (0, 1): np.asarray([0.0, 0.0]),
        (0, 2): np.asarray([1.0, 0.0]),
        (0, 3): np.asarray([0.0, 1.0]),
        (0, 4): np.asarray([1.0, 1.0]),
    }
    target = {
        (1, 11): np.asarray([2.0, 3.0]),
        (1, 12): np.asarray([3.0, 3.0]),
        (1, 13): np.asarray([2.0, 4.0]),
        (1, 14): np.asarray([3.0, 4.0]),
    }

    def fake_centroid(_sessions: object, session: int, roi: int) -> np.ndarray | None:
        return {**source, **target}.get((session, roi))

    monkeypatch.setattr(growth_audit, "_centroid_xy", fake_centroid)

    model = growth_audit._fit_growth_model(
        [],
        (
            (0, 1, 1, 11),
            (0, 1, 2, 12),
            (0, 1, 3, 13),
            (0, 1, 4, 14),
        ),
    )

    predicted = growth_audit._apply_affine(np.asarray([0.25, 0.50]), model.affine_xy)
    assert predicted == pytest.approx(np.asarray([2.25, 3.50]))
    assert model.anchor_count == 4
    assert model.inlier_count == 4


def test_fit_growth_model_uses_translation_fallback_for_two_anchors(
    monkeypatch,
) -> None:
    centroids = {
        (0, 1): np.asarray([2.0, 5.0]),
        (0, 2): np.asarray([8.0, 9.0]),
        (1, 11): np.asarray([3.0, 7.0]),
        (1, 12): np.asarray([9.0, 11.0]),
    }

    def fake_centroid(_sessions: object, session: int, roi: int) -> np.ndarray | None:
        return centroids.get((session, roi))

    monkeypatch.setattr(growth_audit, "_centroid_xy", fake_centroid)

    model = growth_audit._fit_growth_model([], ((0, 1, 1, 11), (0, 1, 2, 12)))

    assert model.model_type == "translation_fallback"
    assert growth_audit._apply_affine(np.asarray([4.0, 6.0]), model.affine_xy) == (
        pytest.approx(np.asarray([5.0, 8.0]))
    )


def test_cosine_handles_zero_vectors() -> None:
    assert np.isnan(growth_audit._cosine(np.asarray([0.0, 0.0]), np.ones(2)))
    assert growth_audit._cosine(np.asarray([1.0, 0.0]), np.asarray([2.0, 0.0])) == (
        pytest.approx(1.0)
    )


def test_motion_context_accepts_single_track_vector(monkeypatch) -> None:
    centroids = {
        (0, 10): np.asarray([0.0, 0.0]),
        (1, 11): np.asarray([1.0, 0.0]),
        (2, 12): np.asarray([2.0, 0.0]),
    }

    def fake_centroid(_sessions: object, session: int, roi: int) -> np.ndarray | None:
        return centroids.get((session, roi))

    monkeypatch.setattr(growth_audit, "_centroid_xy", fake_centroid)

    features = growth_audit._motion_context_features(
        [],
        (1, 2, 11, 12),
        np.asarray([10, 11, 12]),
        np.asarray([1.0, 0.0]),
        np.asarray([2.0, 0.0]),
    )

    assert features["two_edge_motion_consistency"] == pytest.approx(1.0)
    assert features["two_edge_acceleration"] == pytest.approx(0.0)


def test_edge_growth_features_passes_prediction_matrix_to_motion_context(
    monkeypatch,
) -> None:
    predicted = np.asarray([[10, 11, 12]], dtype=int)
    centroids = {
        (0, 10): np.asarray([0.0, 0.0]),
        (1, 11): np.asarray([1.0, 0.0]),
    }
    captured: dict[str, object] = {}

    def fake_centroid(_sessions: object, session: int, roi: int) -> np.ndarray | None:
        return centroids.get((session, roi))

    def fake_motion_context(
        _sessions: object,
        _edge: object,
        predicted_arg: object,
        _source: object,
        _target: object,
    ) -> dict[str, float]:
        captured["predicted"] = predicted_arg
        return {
            "two_edge_motion_consistency": 0.0,
            "two_edge_acceleration": 0.0,
        }

    monkeypatch.setattr(growth_audit, "_centroid_xy", fake_centroid)
    monkeypatch.setattr(growth_audit, "_motion_context_features", fake_motion_context)
    monkeypatch.setattr(growth_audit, "_radial_direction_cosine", lambda *args: 0.0)

    growth_audit._edge_growth_features(
        [],
        (0, 1, 10, 11),
        model=growth_audit._identity_growth_model(),
        anchor_edges=(),
        predicted=predicted,
    )

    assert captured["predicted"] is predicted


def test_pad_track_matrix_fills_missing_sessions_with_minus_one() -> None:
    padded = growth_audit._pad_track_matrix(np.asarray([10, 11]), width=4)

    assert padded.tolist() == [[10, 11, -1, -1]]
