from __future__ import annotations

from bayescatrack.experiments import registration_qa_report
from bayescatrack.experiments import track2p_teacher_audit

# pylint: disable=protected-access


def test_registration_qa_cli_keeps_suite2p_non_cells_by_default():
    parser = registration_qa_report.build_arg_parser()

    default_args = parser.parse_args(["--data", "dataset"])
    assert default_args.include_non_cells is True

    hard_filter_args = parser.parse_args(["--data", "dataset", "--no-include-non-cells"])
    assert hard_filter_args.include_non_cells is False

    config = registration_qa_report._config_from_args(default_args)
    assert config.include_non_cells is True


def test_teacher_audit_cli_keeps_suite2p_non_cells_by_default():
    parser = track2p_teacher_audit.build_arg_parser()

    default_args = parser.parse_args(["--data", "dataset"])
    assert default_args.include_non_cells is True

    hard_filter_args = parser.parse_args(["--data", "dataset", "--no-include-non-cells"])
    assert hard_filter_args.include_non_cells is False

    config = track2p_teacher_audit._config(default_args)
    assert config.include_non_cells is True
