from __future__ import annotations

import pytest
from bayescatrack.experiments import track2p_benchmark


def test_auto_registration_candidates_preserve_valid_tokens() -> None:
    parser = track2p_benchmark.build_arg_parser()
    args = parser.parse_args(
        [
            "--data",
            "/tmp/track2p",
            "--method",
            "global-assignment",
            "--auto-registration-candidates",
            "none, fov-translation,affine",
        ]
    )

    config = track2p_benchmark._config_from_args(args)  # pylint: disable=protected-access

    assert config.auto_registration_candidates == (
        "none",
        "fov-translation",
        "affine",
    )


@pytest.mark.parametrize(
    "raw_value",
    ["", " ", "none,,affine", ",none", "none,"],
)
def test_auto_registration_candidates_reject_empty_tokens(raw_value: str) -> None:
    parser = track2p_benchmark.build_arg_parser()
    args = parser.parse_args(
        [
            "--data",
            "/tmp/track2p",
            "--method",
            "global-assignment",
            "--auto-registration-candidates",
            raw_value,
        ]
    )

    with pytest.raises(
        ValueError,
        match="--auto-registration-candidates must be a comma-separated list of non-empty values",
    ):
        track2p_benchmark._config_from_args(args)  # pylint: disable=protected-access
