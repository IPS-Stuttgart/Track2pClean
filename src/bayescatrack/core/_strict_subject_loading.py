"""Strict subject-loading support for public summary/export helpers.

The lower-level Track2p subject loader already supports ``strict=True`` so callers
can fail on recognized sessions that lack the requested plane. The public
summary/export helpers and their CLI handlers need to propagate that flag as
well; otherwise a strict data-preparation workflow can still silently summarize
or export an incomplete subject.
"""

from __future__ import annotations

import argparse
import json
from functools import wraps
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

# pylint: disable=protected-access,too-many-arguments,too-many-locals

_PATCH_MARKER = "_bayescatrack_strict_subject_loading_patch"
_STRICT_HELP = "Raise instead of skipping recognized sessions that lack the requested plane"


def install_strict_subject_loading(bridge_impl: ModuleType) -> None:
    """Install idempotent strict-loading support on public summary/export paths."""

    if getattr(getattr(bridge_impl, "summarize_subject", None), _PATCH_MARKER, False):
        return

    original_summarize_subject = bridge_impl.summarize_subject
    original_export_subject_to_npz = bridge_impl.export_subject_to_npz
    original_build_arg_parser = bridge_impl._build_arg_parser
    original_handle_summary = bridge_impl._handle_summary
    original_handle_export = bridge_impl._handle_export

    @wraps(original_summarize_subject)
    def summarize_subject(
        subject_dir: str | Path,
        *,
        plane_name: str = "plane0",
        input_format: str = "auto",
        include_behavior: bool = True,
        strict: bool = False,
        **suite2p_kwargs: Any,
    ) -> dict[str, Any]:
        sessions = bridge_impl.load_track2p_subject(
            subject_dir,
            plane_name=plane_name,
            input_format=input_format,
            include_behavior=include_behavior,
            strict=strict,
            **suite2p_kwargs,
        )
        return _summarize_loaded_sessions(
            subject_dir,
            plane_name=plane_name,
            input_format=input_format,
            sessions=sessions,
        )

    @wraps(original_export_subject_to_npz)
    def export_subject_to_npz(
        subject_dir: str | Path,
        output_path: str | Path,
        *,
        plane_name: str = "plane0",
        input_format: str = "auto",
        include_behavior: bool = True,
        include_masks: bool = False,
        order: str = "xy",
        weighted: bool = False,
        velocity_variance: float = 25.0,
        regularization: float = 1e-6,
        validate_pyrecest: bool = False,
        strict: bool = False,
        **suite2p_kwargs: Any,
    ) -> dict[str, Any]:
        sessions = bridge_impl.load_track2p_subject(
            subject_dir,
            plane_name=plane_name,
            input_format=input_format,
            include_behavior=include_behavior,
            strict=strict,
            **suite2p_kwargs,
        )
        return _export_loaded_sessions_to_npz(
            subject_dir,
            output_path,
            plane_name=plane_name,
            input_format=input_format,
            sessions=sessions,
            include_masks=include_masks,
            order=order,
            weighted=weighted,
            velocity_variance=velocity_variance,
            regularization=regularization,
            validate_pyrecest=validate_pyrecest,
        )

    @wraps(original_build_arg_parser)
    def _build_arg_parser(*args: Any, **kwargs: Any) -> argparse.ArgumentParser:
        parser = original_build_arg_parser(*args, **kwargs)
        _add_strict_option_to_subject_parsers(parser)
        return parser

    def _handle_summary(args: argparse.Namespace) -> int:
        summary = bridge_impl.summarize_subject(
            args.subject_dir,
            plane_name=args.plane_name,
            input_format=args.input_format,
            include_behavior=args.include_behavior,
            strict=getattr(args, "strict", False),
            **bridge_impl._suite2p_kwargs_from_args(args),
        )
        print(json.dumps(summary, indent=2))
        return 0

    def _handle_export(args: argparse.Namespace) -> int:
        summary = bridge_impl.export_subject_to_npz(
            args.subject_dir,
            args.output_path,
            plane_name=args.plane_name,
            input_format=args.input_format,
            include_behavior=args.include_behavior,
            include_masks=args.include_masks,
            order=args.order,
            weighted=args.weighted,
            velocity_variance=args.velocity_variance,
            regularization=args.regularization,
            validate_pyrecest=args.validate_pyrecest,
            strict=getattr(args, "strict", False),
            **bridge_impl._suite2p_kwargs_from_args(args),
        )
        print(json.dumps(summary, indent=2))
        return 0

    for patched, original in (
        (summarize_subject, original_summarize_subject),
        (export_subject_to_npz, original_export_subject_to_npz),
        (_build_arg_parser, original_build_arg_parser),
        (_handle_summary, original_handle_summary),
        (_handle_export, original_handle_export),
    ):
        setattr(patched, _PATCH_MARKER, True)
        setattr(patched, "_bayescatrack_original", original)

    bridge_impl.summarize_subject = summarize_subject
    bridge_impl.export_subject_to_npz = export_subject_to_npz
    bridge_impl._build_arg_parser = _build_arg_parser
    bridge_impl._handle_summary = _handle_summary
    bridge_impl._handle_export = _handle_export


def _summarize_loaded_sessions(
    subject_dir: str | Path,
    *,
    plane_name: str,
    input_format: str,
    sessions: list[Any],
) -> dict[str, Any]:
    return {
        "subject_dir": str(Path(subject_dir)),
        "plane_name": plane_name,
        "input_format": input_format,
        "n_sessions": len(sessions),
        "sessions": [
            {
                "session_name": session.session_name,
                "session_date": (
                    session.session_date.isoformat() if session.session_date else None
                ),
                "source": session.plane_data.source,
                "n_rois": session.plane_data.n_rois,
                "image_shape": list(session.plane_data.image_shape),
                "trace_shape": (
                    list(session.plane_data.traces.shape)
                    if session.plane_data.traces is not None
                    else None
                ),
                "has_fov": session.plane_data.fov is not None,
                "has_motion_energy": session.motion_energy is not None,
            }
            for session in sessions
        ],
    }


def _export_loaded_sessions_to_npz(
    subject_dir: str | Path,
    output_path: str | Path,
    *,
    plane_name: str,
    input_format: str,
    sessions: list[Any],
    include_masks: bool,
    order: str,
    weighted: bool,
    velocity_variance: float,
    regularization: float,
    validate_pyrecest: bool,
) -> dict[str, Any]:
    payload: dict[str, np.ndarray] = {
        "session_names": np.asarray(
            [session.session_name for session in sessions], dtype=np.str_
        ),
        "session_dates": np.asarray(
            [
                session.session_date.isoformat() if session.session_date is not None else ""
                for session in sessions
            ],
            dtype=np.str_,
        ),
        "plane_name": np.asarray(str(plane_name), dtype=np.str_),
        "input_format": np.asarray(str(input_format), dtype=np.str_),
    }

    summary_sessions: list[dict[str, Any]] = []
    for session_index, session in enumerate(sessions):
        plane_data = session.plane_data
        export = plane_data.to_export_dict(
            order=order,
            weighted=weighted,
            velocity_variance=velocity_variance,
            regularization=regularization,
            include_masks=include_masks,
        )
        for key, value in export.items():
            payload[f"session_{session_index}__{key}"] = value
        if session.motion_energy is not None:
            payload[f"session_{session_index}__motion_energy"] = session.motion_energy

        if validate_pyrecest:
            _ = plane_data.to_pyrecest_gaussian_distributions(
                order=order,
                weighted=weighted,
                velocity_variance=velocity_variance,
                regularization=regularization,
            )

        summary_sessions.append(
            {
                "session_name": session.session_name,
                "session_date": (
                    session.session_date.isoformat() if session.session_date else None
                ),
                "source": plane_data.source,
                "n_rois": plane_data.n_rois,
                "image_shape": list(plane_data.image_shape),
                "has_traces": plane_data.traces is not None,
                "has_fov": plane_data.fov is not None,
                "has_motion_energy": session.motion_energy is not None,
            }
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)

    return {
        "subject_dir": str(Path(subject_dir)),
        "output_path": str(output_path),
        "n_sessions": len(sessions),
        "plane_name": plane_name,
        "input_format": input_format,
        "sessions": summary_sessions,
    }


def _add_strict_option_to_subject_parsers(parser: argparse.ArgumentParser) -> None:
    for command_name, command_parser in _subcommand_parsers(parser).items():
        if command_name not in {"summary", "export"}:
            continue
        if _has_option(command_parser, "--strict"):
            continue
        command_parser.add_argument(
            "--strict",
            action="store_true",
            help=_STRICT_HELP,
        )


def _subcommand_parsers(
    parser: argparse.ArgumentParser,
) -> dict[str, argparse.ArgumentParser]:
    for action in parser._actions:
        choices = getattr(action, "choices", None)
        if not isinstance(choices, dict):
            continue
        return {
            name: choice
            for name, choice in choices.items()
            if isinstance(choice, argparse.ArgumentParser)
        }
    return {}


def _has_option(parser: argparse.ArgumentParser, option: str) -> bool:
    return any(option in action.option_strings for action in parser._actions)


__all__ = ["install_strict_subject_loading"]
