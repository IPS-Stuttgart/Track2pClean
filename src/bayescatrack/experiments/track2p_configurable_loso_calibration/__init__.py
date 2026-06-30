"""Strict CLI wrapper for the configurable LOSO calibration module.

This package preserves the historical
``bayescatrack.experiments.track2p_configurable_loso_calibration`` import path
while tightening explicit calibration-feature CLI parsing.  Empty tokens in
``--calibration-features`` and empty repeated ``--calibration-feature`` values
are treated as malformed explicit overrides instead of falling back to the named
feature preset.
"""

from __future__ import annotations

import importlib.util as _importlib_util
from pathlib import Path as _Path

_IMPL_PATH = _Path(__file__).resolve().parent.parent / "track2p_configurable_loso_calibration.py"
_SPEC = _importlib_util.spec_from_file_location(f"{__name__}._impl", _IMPL_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import-system guard
    raise ImportError(f"Could not load configurable LOSO calibration implementation from {_IMPL_PATH}")
_IMPL = _importlib_util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_IMPL)

for _name in dir(_IMPL):
    if _name.startswith("__") and _name not in {"__doc__"}:
        continue
    globals()[_name] = getattr(_IMPL, _name)

from .._configurable_loso_feature_cli_validation import (  # noqa: E402
    install_configurable_loso_feature_cli_validation as _install_feature_cli_validation,
)

_install_feature_cli_validation()


def main(argv: list[str] | None = None) -> int:
    """Run the configurable LOSO CLI with strict feature-list validation."""

    args = build_arg_parser().parse_args(argv)
    config = _config_from_args(args)
    result = run_track2p_configurable_loso_calibration(
        config,
        feature_names=_resolved_calibration_feature_names(args),
        sample_weight_strategy=args.sample_weight_strategy,
        model_kind=args.calibration_model,
        model_kwargs=_json_object(
            args.calibration_model_kwargs_json,
            "--calibration-model-kwargs-json",
        ),
        hard_negative_options=_hard_negative_options(args),
    )
    rows = result.to_rows()
    if args.output is None:
        _write_stdout(rows, args.format)
    else:
        write_results(rows, args.output, args.format)
    return 0
