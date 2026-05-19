"""BayesCaTrack command line entry point."""

from __future__ import annotations

import argparse
import sys

from bayescatrack.core.bridge import main as _core_main

_TOP_LEVEL_HELP = """usage: bayescatrack {summary,export,benchmark,growth} ...

BayesCaTrack command line tools.

commands:
  summary      Print a JSON summary for one Track2p-style subject directory.
  export       Export a PyRecEst-ready NPZ bundle for one subject.
  benchmark    Run reproducible benchmark harnesses.
  growth       Analyze global growth/displacement patterns from track tables.

Run 'bayescatrack <command> --help' for command-specific options.
"""


def main(argv: list[str] | None = None) -> int:
    """Dispatch BayesCaTrack CLI commands."""

    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(_TOP_LEVEL_HELP)
        return 0

    if args[0] == "growth":
        from bayescatrack.analysis.growth import main as _growth_main

        return int(_growth_main(args[1:]))

    if args[0] != "benchmark":
        return int(_core_main(args))

    return _handle_benchmark(args[1:])


def _handle_benchmark(args: list[str]) -> int:
    if not args or args[0] in {"-h", "--help"}:
        parser = argparse.ArgumentParser(
            prog="bayescatrack benchmark",
            description="Run BayesCaTrack benchmark harnesses.",
        )
        subparsers = parser.add_subparsers(dest="benchmark", required=False)
        subparsers.add_parser(
            "track2p", help="Track2p baseline and global-assignment ablations"
        )
        subparsers.add_parser(
            "track2p-sweep",
            help="Sweep Track2p global-assignment cost scales and thresholds",
        )
        subparsers.add_parser(
            "track2p-shifted-iou",
            help="Run one Track2p shifted-IoU global-assignment benchmark",
        )
        subparsers.add_parser(
            "track2p-shifted-iou-ablation",
            help="Run a multi-radius shifted-IoU benchmark and edge-ranking ablation",
        )
        subparsers.add_parser(
            "track2p-solver-prior-loso",
            help="Tune Track2p global-assignment solver priors inside LOSO folds",
        )
        subparsers.add_parser(
            "track2p-solver-oracles",
            help="Run solver-oracle global-assignment diagnostics and paper artifacts",
        )
        subparsers.add_parser(
            "track2p-monotone-loso",
            help="Run LOSO calibrated global assignment with monotone ranking costs",
        )
        subparsers.add_parser(
            "track2p-teacher-distill",
            help="Train a monotone ranker from Track2p teacher edges in LOSO folds",
        )
        subparsers.add_parser(
            "track2p-teacher-audit",
            help="Cross-tab manual GT, Track2p output, and BayesCaTrack edges",
        )
        subparsers.add_parser(
            "track2p-teacher-debug",
            help="Export Bayes/Track2p/manual-GT disagreement diagnostics",
        )
        subparsers.add_parser(
            "track2p-teacher-diagnostics",
            help="Alias for track2p-teacher-debug",
        )
        subparsers.add_parser(
            "track2p-teacher-distill-loso",
            help="Train LOSO calibrated costs from Track2p teacher labels and evaluate on manual GT",
        )
        subparsers.add_parser(
            "edge-ranking",
            help="Rank manual-GT Track2p edges within pairwise cost/feature matrices",
        )
        subparsers.add_parser(
            "track2p-learned-edge-ranking",
            help=(
                "Rank manual-GT Track2p edges under LOSO calibrated/monotone learned scores"
            ),
        )
        subparsers.add_parser(
            "solver-oracles",
            help="Run GT edge-cost, rank-k, and oracle-registration solver diagnostics",
        )
        subparsers.add_parser(
            "registration-qa",
            help="Report registration quality on manual-GT Track2p links",
        )
        subparsers.add_parser(
            "oracle-affine-qa",
            help="Compare baseline registration to manual-GT oracle affine geometry",
        )
        subparsers.add_parser(
            "growth-registration-qa",
            help="Report spatially resolved growth/deformation registration QA",
        )
        subparsers.add_parser(
            "nonrigid-backend-audit",
            help="Report nonrigid registration backend diagnostics",
        )
        subparsers.add_parser(
            "validate-track2p-inputs",
            help="Validate manual-GT ROI coverage before Track2p benchmarks",
        )
        subparsers.add_parser(
            "audit-manual-gt-rois",
            help="Audit manual-GT ROI index spaces before Track2p benchmarks",
        )
        subparsers.add_parser(
            "audit-manual-gt-roi-index-space",
            help="Alias for audit-manual-gt-rois",
        )
        subparsers.add_parser(
            "compare", help="Aggregate benchmark CSVs into a comparison table"
        )
        subparsers.add_parser("suite", help="Run a JSON benchmark manifest")
        parser.print_help()
        return 0

    if args[0] == "track2p":
        from bayescatrack.experiments.track2p_benchmark import (
            main as _track2p_benchmark_main,
        )

        return int(_track2p_benchmark_main(args[1:]))
    if args[0] == "track2p-sweep":
        from bayescatrack.experiments.track2p_cost_sweep import (
            main as _track2p_cost_sweep_main,
        )

        return int(_track2p_cost_sweep_main(args[1:]))
    if args[0] == "track2p-shifted-iou":
        from bayescatrack.experiments.track2p_shifted_iou_benchmark import (
            main as _track2p_shifted_iou_main,
        )

        return int(_track2p_shifted_iou_main(args[1:]))
    if args[0] == "track2p-shifted-iou-ablation":
        from bayescatrack.experiments.track2p_shifted_iou_ablation import (
            main as _track2p_shifted_iou_ablation_main,
        )

        return int(_track2p_shifted_iou_ablation_main(args[1:]))
    if args[0] == "track2p-solver-prior-loso":
        if _solver_prior_uses_calibrated_cost(args[1:]):
            from bayescatrack.experiments.track2p_solver_prior_tuning import (
                main as _track2p_solver_prior_loso_main,
            )
        else:
            from bayescatrack.experiments.solver_prior_tuning import (
                main as _track2p_solver_prior_loso_main,
            )

        return int(_track2p_solver_prior_loso_main(args[1:]))
    if args[0] == "track2p-solver-oracles":
        from bayescatrack.experiments.track2p_solver_oracle_benchmark import (
            main as _track2p_solver_oracles_main,
        )

        return int(_track2p_solver_oracles_main(args[1:]))
    if args[0] == "track2p-monotone-loso":
        from bayescatrack.experiments.track2p_monotone_loso_calibration import (
            main as _track2p_monotone_loso_main,
        )

        return int(_track2p_monotone_loso_main(args[1:]))
    if args[0] == "track2p-teacher-distill":
        from bayescatrack.experiments.track2p_teacher_distillation import (
            main as _track2p_teacher_distillation_main,
        )

        return int(_track2p_teacher_distillation_main(args[1:]))
    if args[0] == "track2p-teacher-audit":
        from bayescatrack.experiments.track2p_teacher_audit import (
            main as _track2p_teacher_audit_main,
        )

        return int(_track2p_teacher_audit_main(args[1:]))
    if args[0] in {"track2p-teacher-debug", "track2p-teacher-diagnostics"}:
        from bayescatrack.experiments.track2p_teacher_debug import (
            main as _track2p_teacher_debug_main,
        )

        return int(_track2p_teacher_debug_main(args[1:]))
    if args[0] == "track2p-teacher-distill-loso":
        from bayescatrack.experiments.track2p_teacher_distillation import (
            main as _track2p_teacher_distillation_main,
        )

        return int(_track2p_teacher_distillation_main(args[1:]))
    if args[0] == "edge-ranking":
        from bayescatrack.experiments.track2p_edge_ranking import (
            main as _track2p_edge_ranking_main,
        )

        return int(_track2p_edge_ranking_main(args[1:]))
    if args[0] == "track2p-learned-edge-ranking":
        from bayescatrack.experiments.track2p_learned_edge_ranking import (
            main as _track2p_learned_edge_ranking_main,
        )

        return int(_track2p_learned_edge_ranking_main(args[1:]))
    if args[0] == "solver-oracles":
        from bayescatrack.experiments.track2p_solver_oracles import (
            main as _track2p_solver_oracles_main,
        )

        return int(_track2p_solver_oracles_main(args[1:]))
    if args[0] == "registration-qa":
        from bayescatrack.experiments.registration_qa_report import (
            main as _registration_qa_main,
        )

        return int(_registration_qa_main(args[1:]))
    if args[0] == "oracle-affine-qa":
        from bayescatrack.experiments.oracle_affine_registration_qa import (
            main as _oracle_affine_qa_main,
        )

        return int(_oracle_affine_qa_main(args[1:]))
    if args[0] == "growth-registration-qa":
        from bayescatrack.experiments.growth_registration_qa import (
            main as _growth_registration_qa_main,
        )

        return int(_growth_registration_qa_main(args[1:]))
    if args[0] == "nonrigid-backend-audit":
        from bayescatrack.experiments.nonrigid_backend_audit import (
            main as _nonrigid_backend_audit_main,
        )

        return int(_nonrigid_backend_audit_main(args[1:]))
    if args[0] == "validate-track2p-inputs":
        from bayescatrack.experiments.track2p_input_validator import (
            main as _track2p_input_validator_main,
        )

        return int(_track2p_input_validator_main(args[1:]))
    if args[0] in {"audit-manual-gt-rois", "audit-manual-gt-roi-index-space"}:
        from bayescatrack.experiments.track2p_roi_index_audit import (
            main as _track2p_roi_index_audit_main,
        )

        return int(_track2p_roi_index_audit_main(args[1:]))
    if args[0] == "compare":
        from bayescatrack.experiments.benchmark_comparison import (
            main as _benchmark_comparison_main,
        )

        return int(_benchmark_comparison_main(args[1:]))
    if args[0] == "suite":
        from bayescatrack.experiments.benchmark_manifest import (
            main as _benchmark_manifest_main,
        )

        return int(_benchmark_manifest_main(args[1:]))

    parser = argparse.ArgumentParser(prog="bayescatrack benchmark")
    parser.error(f"unknown benchmark {args[0]!r}")
    return 2


def _solver_prior_uses_calibrated_cost(args: list[str]) -> bool:
    """Return whether the solver-prior command requested learned costs.

    The public ``track2p-solver-prior-loso`` subcommand supports both the older
    fixed-cost prior tuner and the newer calibrated/monotone learned-cost tuner.
    Dispatching here keeps existing fixed-cost workflow invocations unchanged
    while exposing ``--cost calibrated`` through the same CLI entry point.
    """

    for index, arg in enumerate(args):
        if arg == "--cost" and index + 1 < len(args):
            return args[index + 1] == "calibrated"
        if arg.startswith("--cost="):
            return arg.split("=", 1)[1] == "calibrated"
    return False


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
