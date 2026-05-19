from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bayescatrack.experiments import track2p_benchmark


def test_track2p_benchmark_parses_registration_kwargs_json(tmp_path: Path) -> None:
    args = track2p_benchmark.build_arg_parser().parse_args(
        [
            "--data",
            str(tmp_path),
            "--method",
            "global-assignment",
            "--registration-kwargs-json",
            '{"grid_shape": [4, 6], "min_tile_size": 16}',
        ]
    )

    config = track2p_benchmark._config_from_args(args)

    assert config.registration_kwargs == {"grid_shape": [4, 6], "min_tile_size": 16}


def test_registration_kwargs_json_must_decode_to_object(tmp_path: Path) -> None:
    args = track2p_benchmark.build_arg_parser().parse_args(
        [
            "--data",
            str(tmp_path),
            "--method",
            "global-assignment",
            "--registration-kwargs-json",
            "[1, 2, 3]",
        ]
    )

    with pytest.raises(ValueError, match="--registration-kwargs-json"):
        track2p_benchmark._config_from_args(args)


def test_solve_configured_global_assignment_forwards_registration_kwargs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    expected_result = object()

    def fake_solver(sessions: object, **kwargs: object) -> object:
        del sessions
        captured.update(kwargs)
        return expected_result

    monkeypatch.setattr(
        track2p_benchmark, "solve_global_assignment_for_sessions", fake_solver
    )
    config = track2p_benchmark.Track2pBenchmarkConfig(
        data=tmp_path,
        method="global-assignment",
        registration_kwargs={"grid_shape": [3, 3]},
    )

    result = track2p_benchmark.solve_configured_global_assignment([], config)

    assert result is expected_result
    assert captured["registration_kwargs"] == {"grid_shape": [3, 3]}


def test_register_plane_pair_forwards_nonrigid_registration_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bayescatrack.track2p_registration as track2p_registration

    captured: dict[str, object] = {}
    registered_plane = object()

    def fake_nonrigid_register(
        reference_plane: object, moving_plane: object, **kwargs: object
    ) -> object:
        del reference_plane, moving_plane
        captured.update(kwargs)
        return SimpleNamespace(registered_measurement_plane=registered_plane)

    monkeypatch.setattr(
        track2p_registration,
        "register_measurement_plane_by_nonrigid_fov",
        fake_nonrigid_register,
    )
    plane = SimpleNamespace(fov=np.zeros((4, 4)), image_shape=(4, 4))

    assert (
        track2p_registration.register_plane_pair(
            plane,
            plane,
            transform_type="bspline",
            registration_kwargs={"grid_shape": [4, 4], "min_tile_size": 8},
        )
        is registered_plane
    )
    assert captured["transform_type"] == "bspline"
    assert captured["grid_shape"] == [4, 4]
    assert captured["min_tile_size"] == 8
