from __future__ import annotations

import pytest

from bayescatrack.experiments import (
    oracle_affine_registration_qa,
    registration_qa_report,
    solver_prior_tuning,
    track2p_benchmark,
    track2p_calibration_export,
    track2p_cost_sweep,
)
from bayescatrack.track2p_registration import REGISTRATION_TRANSFORM_TYPES

# pylint: disable=protected-access

NONRIGID_BENCHMARK_TRANSFORMS = (
    "fov-affine",
    "bspline",
    "tps",
    "local-affine-grid",
    "optical-flow",
)


def _transform_choices(parser):
    for action in parser._actions:
        if "--transform-type" in action.option_strings:
            return tuple(action.choices or ())
    raise AssertionError("parser does not expose --transform-type")


@pytest.mark.parametrize(
    "parser",
    (
        track2p_benchmark.build_arg_parser(),
        track2p_cost_sweep.build_arg_parser(),
        solver_prior_tuning.build_arg_parser(),
        track2p_calibration_export.build_arg_parser(),
        registration_qa_report.build_arg_parser(),
    ),
)
def test_core_track2p_clis_expose_registration_transform_types(parser):
    assert set(REGISTRATION_TRANSFORM_TYPES).issubset(_transform_choices(parser))


@pytest.mark.parametrize("transform_type", NONRIGID_BENCHMARK_TRANSFORMS)
def test_track2p_benchmark_cli_accepts_nonrigid_registration_transform(
    transform_type,
):
    args = track2p_benchmark.build_arg_parser().parse_args(
        [
            "--data",
            "dataset",
            "--method",
            "global-assignment",
            "--transform-type",
            transform_type,
        ]
    )

    config = track2p_benchmark._config_from_args(args)

    assert config.transform_type == transform_type


def test_track2p_benchmark_cli_accepts_fov_translation_transform():
    args = track2p_benchmark.build_arg_parser().parse_args(
        [
            "--data",
            "dataset",
            "--method",
            "global-assignment",
            "--transform-type",
            "fov-translation",
        ]
    )

    config = track2p_benchmark._config_from_args(args)

    assert config.transform_type == "fov-translation"


def test_track2p_benchmark_cli_accepts_growth_transform_without_argparse_patch():
    args = track2p_benchmark.build_arg_parser().parse_args(
        [
            "--data",
            "dataset",
            "--method",
            "global-assignment",
            "--transform-type",
            "bspline",
        ]
    )

    config = track2p_benchmark._config_from_args(args)

    assert config.transform_type == "bspline"


def test_track2p_cost_sweep_cli_accepts_fov_translation_transform():
    args = track2p_cost_sweep.build_arg_parser().parse_args(
        [
            "--data",
            "dataset",
            "--cost-scales",
            "1",
            "--cost-thresholds",
            "6",
            "--transform-type",
            "fov-translation",
        ]
    )

    config = track2p_cost_sweep._config_from_args(args)

    assert config.benchmark.transform_type == "fov-translation"


def test_solver_prior_loso_cli_accepts_fov_translation_transform():
    args = solver_prior_tuning.build_arg_parser().parse_args(
        [
            "--data",
            "dataset",
            "--transform-type",
            "fov-translation",
            "--start-costs",
            "1",
            "--end-costs",
            "1",
            "--gap-penalties",
            "0",
            "--cost-thresholds",
            "none",
        ]
    )

    assert args.transform_type == "fov-translation"


def test_calibration_export_cli_accepts_fov_translation_transform():
    args = track2p_calibration_export.build_arg_parser().parse_args(
        [
            "--data",
            "dataset",
            "--output",
            "calibration.csv",
            "--transform-type",
            "fov-translation",
        ]
    )

    assert args.transform_type == "fov-translation"


def test_registration_qa_cli_accepts_fov_translation_transform():
    args = registration_qa_report.build_arg_parser().parse_args(
        ["--data", "dataset", "--transform-type", "fov-translation"]
    )

    config = registration_qa_report._config_from_args(args)

    assert config.transform_type == "fov-translation"


def test_oracle_affine_qa_cli_accepts_fov_translation_transform():
    args = oracle_affine_registration_qa.build_arg_parser().parse_args(
        ["--data", "dataset", "--transform-type", "fov-translation"]
    )

    config = oracle_affine_registration_qa.OracleAffineQAConfig(
        registration=registration_qa_report._config_from_args(args)
    )

    assert config.registration.transform_type == "fov-translation"


def test_registration_qa_cli_accepts_gt_affine_oracle_transform():
    args = registration_qa_report.build_arg_parser().parse_args(
        ["--data", "dataset", "--transform-type", "gt-affine-oracle"]
    )

    config = registration_qa_report._config_from_args(args)

    assert config.transform_type == "gt-affine-oracle"
