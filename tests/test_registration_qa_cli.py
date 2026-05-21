from __future__ import annotations

from bayescatrack.experiments import registration_qa_report

# pylint: disable=protected-access


def test_registration_qa_cli_keeps_suite2p_non_cells_by_default():
    parser = registration_qa_report.build_arg_parser()

    default_args = parser.parse_args(["--data", "dataset"])
    assert default_args.include_non_cells is True

    hard_filter_args = parser.parse_args(
        ["--data", "dataset", "--no-include-non-cells"]
    )
    assert hard_filter_args.include_non_cells is False

    config = registration_qa_report._config_from_args(default_args)
    assert config.include_non_cells is True
