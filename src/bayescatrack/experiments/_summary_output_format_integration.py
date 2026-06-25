"""Keep auxiliary summary row exports on the diagnostics output channel."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Literal, cast

RowOutputFormat = Literal["csv", "json"]
SummaryOutputContext = tuple[Path, RowOutputFormat] | None
_SUMMARY_OUTPUT_CONTEXT: ContextVar[SummaryOutputContext] = ContextVar(
    "_SUMMARY_OUTPUT_CONTEXT",
    default=None,
)
_MARKER = "_bayescatrack_summary_output_format_context"


def install_summary_output_format_integration() -> None:
    """Patch summary-row writers to use ``--diagnostics-format``.

    Several residual-cleanup CLIs write their benchmark table/CSV/JSON via
    ``--format`` but write auxiliary row ledgers via the shared diagnostics
    writer.  Their summary rows are also auxiliary row ledgers, so the summary
    path must follow ``--diagnostics-format`` rather than the primary benchmark
    result format.
    """

    from bayescatrack.experiments import track2p_policy_growth_field_residual_audit
    from bayescatrack.experiments import track2p_policy_growth_veto_cleanup
    from bayescatrack.experiments import track2p_policy_growth_veto_whatif

    _patch_cleanup_parser_factory(track2p_policy_growth_veto_cleanup)
    patched_write_rows = _patched_write_rows(
        track2p_policy_growth_field_residual_audit.write_rows
    )
    track2p_policy_growth_field_residual_audit.write_rows = patched_write_rows
    track2p_policy_growth_veto_whatif.write_rows = patched_write_rows


def _patch_cleanup_parser_factory(module: Any) -> None:
    original_build_arg_parser = module.build_arg_parser
    if getattr(original_build_arg_parser, _MARKER, False):
        return

    def _build_arg_parser_with_summary_context() -> argparse.ArgumentParser:
        parser = original_build_arg_parser()
        _patch_parser_parse_args(parser)
        return parser

    setattr(_build_arg_parser_with_summary_context, _MARKER, True)
    setattr(_build_arg_parser_with_summary_context, "_bayescatrack_original", original_build_arg_parser)
    module.build_arg_parser = _build_arg_parser_with_summary_context


def _patch_parser_parse_args(parser: argparse.ArgumentParser) -> None:
    original_parse_args = parser.parse_args
    if getattr(original_parse_args, _MARKER, False):
        return

    def _parse_args_with_summary_context(
        args: Sequence[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        parsed = original_parse_args(args, namespace)
        _capture_summary_output_context(parsed)
        return parsed

    setattr(_parse_args_with_summary_context, _MARKER, True)
    setattr(_parse_args_with_summary_context, "_bayescatrack_original", original_parse_args)
    parser.parse_args = _parse_args_with_summary_context


def _capture_summary_output_context(args: argparse.Namespace) -> None:
    summary_output = getattr(args, "summary_output", None)
    diagnostics_format = getattr(args, "diagnostics_format", None)
    if summary_output is None or diagnostics_format not in {"csv", "json"}:
        _SUMMARY_OUTPUT_CONTEXT.set(None)
        return
    _SUMMARY_OUTPUT_CONTEXT.set(
        (Path(summary_output), cast(RowOutputFormat, diagnostics_format))
    )


def _patched_write_rows(original_write_rows: Any) -> Any:
    if getattr(original_write_rows, _MARKER, False):
        return original_write_rows

    def _write_rows_with_summary_context(
        rows: Sequence[Mapping[str, Any]],
        output_path: Path,
        *,
        output_format: RowOutputFormat = "csv",
    ) -> None:
        selected_format: str = str(output_format)
        context = _SUMMARY_OUTPUT_CONTEXT.get()
        if context is not None:
            summary_output, summary_format = context
            if Path(output_path) == summary_output:
                selected_format = summary_format
        original_write_rows(
            rows,
            output_path,
            output_format=cast(RowOutputFormat, selected_format),
        )

    setattr(_write_rows_with_summary_context, _MARKER, True)
    setattr(_write_rows_with_summary_context, "_bayescatrack_original", original_write_rows)
    return _write_rows_with_summary_context
