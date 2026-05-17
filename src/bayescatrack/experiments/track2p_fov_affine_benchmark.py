"""Track2p benchmark wrapper using BayesCaTrack's FOV-affine registration."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from bayescatrack.core.bridge import CalciumPlaneData
from bayescatrack.experiments.track2p_benchmark import (
    _config_from_args,
    _csv_fieldnames,
    build_arg_parser,
    run_track2p_benchmark,
    write_results,
)
from bayescatrack.fov_affine_registration import register_measurement_plane_by_fov_affine


def _register_plane_pair_with_fov_affine(
    reference_plane: CalciumPlaneData,
    moving_plane: CalciumPlaneData,
    *,
    transform_type: str = "affine",
) -> CalciumPlaneData:
    if transform_type == "fov-affine":
        return register_measurement_plane_by_fov_affine(
            reference_plane,
            moving_plane,
        ).registered_measurement_plane
    from bayescatrack.track2p_registration import register_plane_pair

    return register_plane_pair(
        reference_plane,
        moving_plane,
        transform_type=transform_type,
    )


def _enable_fov_affine_choice(parser: Any) -> None:
    for action in parser._actions:  # pylint: disable=protected-access
        if action.dest == "transform_type":
            action.choices = ("affine", "rigid", "fov-affine", "fov-translation", "none")
            action.default = "fov-affine"
            action.help = "Registration transform; fov-affine is the default for this wrapper"
            return
    raise RuntimeError("Could not find --transform-type action")


def _write_stdout(rows: list[dict[str, Any]], output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(rows, indent=2))
        return
    if output_format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=_csv_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)
        return
    from bayescatrack.experiments.track2p_benchmark import format_benchmark_table

    print(format_benchmark_table(rows))


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    parser.prog = "bayescatrack benchmark track2p-fov-affine"
    _enable_fov_affine_choice(parser)
    args = parser.parse_args(argv)
    config = _config_from_args(args)

    import bayescatrack.association.pyrecest_global_assignment as assignment

    original_register = assignment.register_plane_pair
    assignment.register_plane_pair = _register_plane_pair_with_fov_affine
    try:
        results = run_track2p_benchmark(config)
    finally:
        assignment.register_plane_pair = original_register

    rows = [result.to_dict() for result in results]
    for row in rows:
        row["registration_wrapper"] = "fov-affine"
    if args.output is not None:
        write_results(rows, Path(args.output), args.format)
    else:
        _write_stdout(rows, args.format)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
