"""BayesCaTrack command line entry point."""

from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass

from bayescatrack.core.bridge import main as _core_main

_TOP_LEVEL_HELP = """usage: bayescatrack {summary,export,benchmark,growth,advanced} ...

BayesCaTrack command line tools.

commands:
  summary       Print a JSON summary for one Track2p-style subject directory.
  export        Export a PyRecEst-ready NPZ bundle for one subject.
  benchmark     Run reproducible benchmark harnesses.
  growth        Analyze global growth/displacement patterns from track tables.
  advanced      Advanced diagnostics and result-improvement workbench helpers.

Run 'bayescatrack <command> --help' for command-specific options.
"""


@dataclass(frozen=True)
class _BenchmarkCommand:
    """One benchmark CLI subcommand target."""

    module: str
    help: str


def _cmd(module: str, help_text: str) -> _BenchmarkCommand:
    return _BenchmarkCommand(f"bayescatrack.experiments.{module}", help_text)


_BENCHMARK_COMMANDS: dict[str, _BenchmarkCommand] = {
    "track2p": _cmd("track2p_benchmark", "Track2p baseline and global-assignment ablations"),
    "track2p-policy": _cmd("track2p_policy_benchmark", "Run the first-class Track2p-policy benchmark method"),
    "track2p-policy-audit": _cmd("track2p_policy_audit", "Export a duplicate-aware edge ledger for Track2p-policy results"),
    "track2p-policy-dp": _cmd("track2p_policy_dp_benchmark", "Run the DP-rescued Track2p-policy benchmark method"),
    "track2p-policy-pruned": _cmd("track2p_policy_pruned_benchmark", "Run conservative prune-only Track2p-policy benchmark method"),
    "track2p-policy-component-audit": _cmd("track2p_policy_component_audit", "Audit Track2p-policy components and split weak bridges"),
    "track2p-policy-component-sweep": _cmd("track2p_policy_component_sweep", "Sweep Track2p-policy component-cleanup operating points"),
    "track2p-policy-stability-cleanup": _cmd("track2p_policy_stability_cleanup", "Run prune-only Track2p-policy threshold-stability cleanup"),
    "track2p-policy-multisplit-cleanup": _cmd("track2p_policy_multisplit_cleanup", "Run guarded multi-bridge Track2p-policy component cleanup"),
    "track2p-policy-consensus-cleanup": _cmd("track2p_policy_consensus_cleanup", "Run conservative Track2p-policy consensus bridge cleanup"),
    "track2p-policy-gap-consensus-cleanup": _cmd("track2p_policy_gap_consensus_cleanup", "Run gap-rescue Track2p-policy consensus bridge cleanup"),
    "track2p-shifted-iou": _cmd("track2p_shifted_iou_benchmark", "Track2p global-assignment ablation with residual shifted-IoU costs"),
    "track2p-sweep": _cmd("track2p_cost_sweep", "Sweep Track2p global-assignment cost scales and thresholds"),
    "track2p-search": _cmd("track2p_experiment_search", "Run a compact grid search over Track2p global-assignment protocols"),
    "track2p-oracle-variants": _cmd("track2p_oracle_variants", "Score reference-row, consecutive-link and gap-limited oracle variants"),
    "track2p-error-taxonomy": _cmd("track2p_error_taxonomy", "Classify prediction false-positive and false-negative links"),
    "track2p-activity-tie-breaker-sweep": _cmd("track2p_activity_tie_breaker_sweep", "Sweep weak activity tie-breaker weights for Track2p global assignment"),
    "track2p-mask-input-sweep": _cmd("track2p_mask_input_sweep", "Sweep Suite2p ROI filtering, weighted masks, and overlap-pixel handling"),
    "track2p-solver-prior-loso": _cmd("solver_prior_tuning", "Tune Track2p global-assignment solver priors inside LOSO folds"),
    "track2p-calibrated-solver-prior-loso": _cmd("track2p_solver_prior_tuning", "Run calibrated LOSO global assignment with fold-internal solver-prior tuning"),
    "track2p-loso-calibration": _cmd("track2p_configurable_loso_calibration", "Run configurable hard-negative LOSO calibrated global assignment"),
    "track2p-monotone-loso": _cmd("track2p_monotone_loso_calibration", "Run LOSO calibrated global assignment with monotone ranking costs"),
    "track2p-result-improvement": _cmd("track2p_result_improvement_selection", "Run the Track2p improvement suite and select by complete-track objective"),
    "track2p-teacher-audit": _cmd("track2p_teacher_audit", "Cross-tab manual GT, Track2p output, and BayesCaTrack edges"),
    "track2p-teacher-debug": _cmd("track2p_teacher_debug", "Export Bayes/Track2p/manual-GT disagreement diagnostics"),
    "track2p-diagnose": _cmd("track2p_failure_diagnosis", "Triage Track2p result failures into registration, ranking, solver, or scoring fixes"),
    "edge-ranking": _cmd("track2p_edge_ranking", "Rank manual-GT Track2p edges within pairwise cost/feature matrices"),
    "select-edge-ranking-features": _cmd("edge_ranking_feature_selection", "Select calibrated feature names from edge-ranking summary CSVs"),
    "select-structured-objective": _cmd("structured_objective_tuning", "Rank benchmark variants by complete-track or other structured metrics"),
    "registration-qa": _cmd("registration_qa_report", "Report registration quality on manual-GT Track2p links"),
    "oracle-affine-qa": _cmd("oracle_affine_registration_qa", "Compare baseline registration to manual-GT oracle affine geometry"),
    "growth-registration-qa": _cmd("growth_registration_qa", "Report spatially resolved growth/deformation registration QA"),
    "validate-track2p-inputs": _cmd("track2p_input_validator", "Validate manual-GT ROI coverage before Track2p benchmarks"),
    "audit-manual-gt-rois": _cmd("track2p_roi_index_audit", "Audit manual-GT ROI index spaces before Track2p benchmarks"),
    "compare": _cmd("benchmark_comparison", "Aggregate benchmark CSVs into a comparison table"),
    "suite": _cmd("benchmark_manifest", "Run a JSON benchmark manifest"),
    "validate-suite": _cmd("benchmark_manifest_plan", "Validate a JSON benchmark manifest without running benchmarks"),
    "resolve-suite": _cmd("benchmark_manifest_resolver", "Resolve benchmark manifest root placeholders into an executable copy"),
}

_BENCHMARK_ALIASES: dict[str, str] = {
    "track2p-component-cleanup": "track2p-policy-component-audit",
    "track2p-component-cleanup-sweep": "track2p-policy-component-sweep",
    "track2p-stability-cleanup": "track2p-policy-stability-cleanup",
    "track2p-multisplit-cleanup": "track2p-policy-multisplit-cleanup",
    "track2p-component-multisplit-cleanup": "track2p-policy-multisplit-cleanup",
    "track2p-consensus-cleanup": "track2p-policy-consensus-cleanup",
    "track2p-component-consensus-cleanup": "track2p-policy-consensus-cleanup",
    "track2p-gap-consensus-cleanup": "track2p-policy-gap-consensus-cleanup",
    "track2p-component-gap-consensus-cleanup": "track2p-policy-gap-consensus-cleanup",
    "track2p-teacher-diagnostics": "track2p-teacher-debug",
    "audit-manual-gt-roi-index-space": "audit-manual-gt-rois",
}


def main(argv: list[str] | None = None) -> int:
    """Dispatch BayesCaTrack CLI commands."""

    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(_TOP_LEVEL_HELP)
        return 0
    if args[0] == "growth":
        from bayescatrack.analysis.growth import main as _growth_main

        return int(_growth_main(args[1:]))
    if args[0] == "advanced":
        from bayescatrack.experiments.advanced_improvement_workbench import main as _advanced_main

        return int(_advanced_main(args[1:]))
    if args[0] != "benchmark":
        return int(_core_main(args))
    return _handle_benchmark(args[1:])


def _handle_benchmark(args: list[str]) -> int:
    if not args or args[0] in {"-h", "--help"}:
        _build_benchmark_help_parser().parse_args(args)
        return 0
    command_name = _BENCHMARK_ALIASES.get(args[0], args[0])
    command = _BENCHMARK_COMMANDS.get(command_name)
    if command is None:
        parser = argparse.ArgumentParser(prog="bayescatrack benchmark")
        parser.error(f"unknown benchmark {args[0]!r}")
        return 2
    module = importlib.import_module(command.module)
    return int(module.main(args[1:]))


def _build_benchmark_help_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark",
        description="Run BayesCaTrack benchmark harnesses.",
    )
    subparsers = parser.add_subparsers(dest="benchmark", required=False)
    for name, command in _BENCHMARK_COMMANDS.items():
        subparsers.add_parser(name, help=command.help)
    for alias, canonical in _BENCHMARK_ALIASES.items():
        subparsers.add_parser(alias, help=f"Alias for {canonical}")
    return parser


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
