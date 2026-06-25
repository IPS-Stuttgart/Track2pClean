from __future__ import annotations

import json
from pathlib import Path

from bayescatrack.experiments import track2p_policy_growth_veto_cleanup as cleanup
from bayescatrack.experiments import track2p_policy_growth_veto_whatif as veto


def _parse_args(tmp_path: Path, *, diagnostics_format: str = "csv"):
    return cleanup.build_arg_parser().parse_args(
        [
            "--data",
            "track2p-root",
            "--output",
            str(tmp_path / "benchmark.json"),
            "--format",
            "json",
            "--summary-output",
            str(tmp_path / "summary.csv"),
            "--diagnostics-format",
            diagnostics_format,
        ]
    )


def test_summary_output_uses_diagnostics_format_not_benchmark_format(tmp_path: Path) -> None:
    args = _parse_args(tmp_path, diagnostics_format="csv")

    veto.write_rows(
        [{"subject": "jm038", "selected": 1}],
        args.summary_output,
        output_format=args.format,
    )

    text = args.summary_output.read_text(encoding="utf-8")
    assert text == "selected,subject\n1,jm038\n"


def test_non_summary_row_output_keeps_requested_format(tmp_path: Path) -> None:
    args = _parse_args(tmp_path, diagnostics_format="csv")
    diagnostics_output = tmp_path / "diagnostics.json"

    veto.write_rows(
        [{"subject": "jm038", "selected": 1}],
        diagnostics_output,
        output_format=args.format,
    )

    assert json.loads(diagnostics_output.read_text(encoding="utf-8")) == [
        {"selected": 1, "subject": "jm038"}
    ]


def test_summary_output_format_context_is_consumed(tmp_path: Path) -> None:
    args = _parse_args(tmp_path, diagnostics_format="csv")

    veto.write_rows(
        [{"subject": "jm038", "selected": 1}],
        args.summary_output,
        output_format=args.format,
    )
    veto.write_rows(
        [{"subject": "jm039", "selected": 2}],
        args.summary_output,
        output_format="json",
    )

    assert json.loads(args.summary_output.read_text(encoding="utf-8")) == [
        {"selected": 2, "subject": "jm039"}
    ]
