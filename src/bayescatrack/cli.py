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


_BENCHMARK_COMMAND_DATA: tuple[tuple[str, str, str], ...] = (
    (
        "track2p",
        "bayescatrack.experiments.track2p_benchmark",
        "Track2p baseline and global-assignment ablations",
    ),
    (
        "track2p-policy",
        "bayescatrack.experiments.track2p_policy_benchmark",
        "Run the first-class Track2p-policy benchmark method",
    ),
    (
        "track2p-policy-audit",
        "bayescatrack.experiments.track2p_policy_audit",
        "Export a duplicate-aware edge ledger for Track2p-policy results",
    ),
    (
        "track2p-policy-dp",
        "bayescatrack.experiments.track2p_policy_dp_benchmark",
        "Run the DP-rescued Track2p-policy benchmark method",
    ),
    (
        "track2p-policy-pruned",
        "bayescatrack.experiments.track2p_policy_pruned_benchmark",
        "Run conservative prune-only Track2p-policy benchmark method",
    ),
    (
        "track2p-policy-gap-pruned",
        "bayescatrack.experiments.track2p_policy_gap_pruned_benchmark",
        "Run Track2p-policy gap rescue plus conservative edge pruning",
    ),
    (
        "track2p-policy-component-audit",
        "bayescatrack.experiments.track2p_policy_component_audit",
        "Audit Track2p-policy components and split weak bridges",
    ),
    (
        "track2p-policy-component-residual-audit",
        "bayescatrack.experiments.track2p_policy_component_residual_audit",
        "Audit residual errors after Track2p-policy component cleanup",
    ),
    (
        "track2p-policy-component-residual-whatif",
        "bayescatrack.experiments.track2p_policy_component_residual_whatif",
        "Rank oracle single-edit what-if repairs for ComponentCleanup residuals",
    ),
    (
        "track2p-policy-seed-sensitivity-audit",
        "bayescatrack.experiments.track2p_policy_seed_sensitivity_audit",
        "Audit ComponentCleanup sensitivity to the seed-session choice",
    ),
    (
        "track2p-policy-fragment-stitch-whatif",
        "bayescatrack.experiments.track2p_policy_fragment_stitch_whatif",
        "Audit minimal oracle stitches for persistent complete-track FNs",
    ),
    (
        "track2p-policy-suffix-stitch-ranking-audit",
        "bayescatrack.experiments.track2p_policy_suffix_stitch_ranking_audit",
        "Rank short suffix-stitch candidates from non-GT features",
    ),
    (
        "track2p-policy-coherence-suffix-stitch-whatif",
        "bayescatrack.experiments.track2p_policy_coherence_suffix_stitch_whatif",
        "Run coherence-gated suffix-stitch what-if after ComponentCleanup",
    ),
    (
        "track2p-policy-coherence-suffix-stitch",
        "bayescatrack.experiments.track2p_policy_coherence_suffix_stitch",
        "Run component cleanup plus coherence-gated suffix stitch",
    ),
    (
        "track2p-policy-coherence-suffix-teacher-rescue",
        "bayescatrack.experiments.track2p_policy_coherence_suffix_teacher_rescue",
        "Run coherence suffix stitch plus Track2p-teacher adjacent rescue",
    ),
    (
        "track2p-policy-coherence-teacher-overlay-audit",
        "bayescatrack.experiments.track2p_policy_coherence_teacher_overlay_audit",
        "Audit Track2p-teacher adjacent edges after coherence suffix stitching",
    ),
    (
        "track2p-policy-growth-field-residual-audit",
        "bayescatrack.experiments.track2p_policy_growth_field_residual_audit",
        "Audit residual errors after teacher rescue with growth-field priors",
    ),
    (
        "track2p-policy-growth-veto-whatif",
        "bayescatrack.experiments.track2p_policy_growth_veto_whatif",
        "Audit growth-veto one-edge removals over accepted teacher-rescue edges",
    ),
    (
        "track2p-policy-growth-veto-cleanup",
        "bayescatrack.experiments.track2p_policy_growth_veto_cleanup",
        "Run CoherenceSuffixTeacherRescue plus a strict label-free growth veto",
    ),
    (
        "track2p-policy-coherence-suffix-growth-veto-cleanup",
        "bayescatrack.experiments.track2p_policy_coherence_suffix_growth_veto_cleanup",
        "Run CoherenceSuffixStitch plus a strict label-free growth veto",
    ),
    (
        "track2p-policy-pyrecest-residual-mht-cleanup",
        "bayescatrack.experiments.track2p_policy_pyrecest_residual_mht_cleanup",
        "Run PyRecEst bounded residual MHT over growth-veto hypotheses",
    ),
    (
        "track2p-policy-pyrecest-frontier-mht-cleanup",
        "bayescatrack.experiments.track2p_policy_pyrecest_frontier_mht_cleanup",
        "Run relaxed/frontier PyRecEst residual MHT over growth-veto hypotheses",
    ),
    (
        "track2p-policy-coherence-suffix-exposure-audit",
        "bayescatrack.experiments.track2p_policy_coherence_suffix_exposure_audit",
        "Audit coherence suffix gate exposure without manual GT labels",
    ),
    (
        "track2p-policy-coherence-pareto-whatif",
        "bayescatrack.experiments.track2p_policy_coherence_pareto_whatif",
        "Run exact one-edit Pareto what-ifs after CoherenceSuffixStitch",
    ),
    (
        "track2p-policy-growth-regularized-assignment",
        "bayescatrack.experiments.track2p_policy_growth_regularized_assignment",
        "Run Track2pPolicy with growth-regularized Hungarian assignment",
    ),
    (
        "track2p-policy-teacher-free-adjacent-rescue-ranking-audit",
        (
            "bayescatrack.experiments."
            "track2p_policy_teacher_free_adjacent_rescue_ranking_audit"
        ),
        "Rank teacher-free adjacent-rescue candidates after CoherenceSuffixGrowthVeto",
    ),
    (
        "track2p-policy-teacher-fn-audit",
        "bayescatrack.experiments.track2p_policy_teacher_fn_audit",
        "Audit Track2p-supported false negatives after component cleanup",
    ),
    (
        "track2p-policy-teacher-adjacent-rescue",
        "bayescatrack.experiments.track2p_policy_teacher_adjacent_rescue",
        "Run component cleanup plus seed-anchored Track2p adjacent rescue",
    ),
    (
        "track2p-policy-teacher-veto-cleanup",
        "bayescatrack.experiments.track2p_policy_teacher_veto_cleanup",
        "Run component cleanup plus conservative Track2p-teacher veto cleanup",
    ),
    (
        "track2p-policy-component-sweep",
        "bayescatrack.experiments.track2p_policy_component_sweep",
        "Sweep Track2p-policy component-cleanup operating points",
    ),
    (
        "track2p-policy-stability-cleanup",
        "bayescatrack.experiments.track2p_policy_stability_cleanup",
        "Run prune-only Track2p-policy threshold-stability cleanup",
    ),
    (
        "track2p-policy-multisplit-cleanup",
        "bayescatrack.experiments.track2p_policy_multisplit_cleanup",
        "Run guarded multi-bridge Track2p-policy component cleanup",
    ),
    (
        "track2p-policy-consensus-cleanup",
        "bayescatrack.experiments.track2p_policy_consensus_cleanup",
        "Run conservative Track2p-policy consensus bridge cleanup",
    ),
    (
        "track2p-policy-gap-component-cleanup",
        "bayescatrack.experiments.track2p_policy_gap_component_cleanup",
        "Run Track2p-policy gap rescue plus weakest-bridge component cleanup",
    ),
    (
        "track2p-policy-strict-gated-gap-cleanup",
        "bayescatrack.experiments.track2p_policy_strict_gated_gap_cleanup",
        "Run component cleanup plus strictly gated gap-rescue candidates",
    ),
    (
        "track2p-policy-confidence-ordered-strict-gated-gap-cleanup",
        "bayescatrack.experiments.track2p_policy_confidence_ordered_strict_gap_cleanup",
        "Run component cleanup plus confidence-ordered strictly gated gap rescue",
    ),
    (
        "track2p-policy-gap-bridge-cleanup",
        "bayescatrack.experiments.track2p_policy_gap_bridge_cleanup",
        "Run Track2p-policy gap rescue plus observed-bridge cleanup",
    ),
    (
        "track2p-policy-gap-edge-audit",
        "bayescatrack.experiments.track2p_policy_gap_edge_audit",
        "Audit gap-rescue candidate edges absent from component cleanup",
    ),
    (
        "track2p-policy-gap-consensus-cleanup",
        "bayescatrack.experiments.track2p_policy_gap_consensus_cleanup",
        "Run Track2p-policy gap rescue plus conservative consensus cleanup",
    ),
    (
        "track2p-policy-gap-consensus-sweep",
        "bayescatrack.experiments.track2p_policy_gap_consensus_sweep",
        "Sweep Track2p-policy gap-rescue consensus cleanup operating points",
    ),
    (
        "track2p-policy-gap-consensus-guarded-sweep",
        "bayescatrack.experiments.track2p_policy_gap_consensus_guarded_sweep",
        "Sweep guarded gap-consensus cleanup while keeping adjacent-only as a candidate",
    ),
    (
        "track2p-shifted-iou",
        "bayescatrack.experiments.track2p_shifted_iou_benchmark",
        "Track2p global-assignment ablation with residual shifted-IoU costs",
    ),
    (
        "track2p-sweep",
        "bayescatrack.experiments.track2p_cost_sweep",
        "Sweep Track2p global-assignment cost scales and thresholds",
    ),
    (
        "track2p-search",
        "bayescatrack.experiments.track2p_experiment_search",
        "Run a compact grid search over Track2p global-assignment protocols",
    ),
    (
        "track2p-oracle-variants",
        "bayescatrack.experiments.track2p_oracle_variants",
        "Score reference-row, consecutive-link and gap-limited oracle variants",
    ),
    (
        "track2p-error-taxonomy",
        "bayescatrack.experiments.track2p_error_taxonomy",
        "Classify prediction false-positive and false-negative links",
    ),
    (
        "track2p-activity-tie-breaker-sweep",
        "bayescatrack.experiments.track2p_activity_tie_breaker_sweep",
        "Sweep weak activity tie-breaker weights for Track2p global assignment",
    ),
    (
        "track2p-mask-input-sweep",
        "bayescatrack.experiments.track2p_mask_input_sweep",
        "Sweep Suite2p ROI filtering, weighted masks, and overlap-pixel handling",
    ),
    (
        "track2p-solver-prior-loso",
        "bayescatrack.experiments.solver_prior_tuning",
        "Tune Track2p global-assignment solver priors inside LOSO folds",
    ),
    (
        "track2p-calibrated-solver-prior-loso",
        "bayescatrack.experiments.track2p_solver_prior_tuning",
        "Run calibrated LOSO global assignment with fold-internal solver-prior tuning",
    ),
    (
        "track2p-loso-calibration",
        "bayescatrack.experiments.track2p_configurable_loso_calibration",
        "Run configurable hard-negative LOSO calibrated global assignment",
    ),
    (
        "track2p-gap-balanced-loso",
        "bayescatrack.experiments.track2p_gap_balanced_loso_calibration",
        "Run LOSO calibrated global assignment with gap-balanced sample weights",
    ),
    (
        "track2p-monotone-loso",
        "bayescatrack.experiments.track2p_monotone_loso_calibration",
        "Run LOSO calibrated global assignment with monotone ranking costs",
    ),
    (
        "track2p-result-improvement",
        "bayescatrack.experiments.track2p_result_improvement_selection",
        "Run the Track2p improvement suite and select by complete-track objective",
    ),
    (
        "track2p-teacher-audit",
        "bayescatrack.experiments.track2p_teacher_audit",
        "Cross-tab manual GT, Track2p output, and BayesCaTrack edges",
    ),
    (
        "track2p-teacher-debug",
        "bayescatrack.experiments.track2p_teacher_debug",
        "Export Bayes/Track2p/manual-GT disagreement diagnostics",
    ),
    (
        "track2p-diagnose",
        "bayescatrack.experiments.track2p_failure_diagnosis",
        "Triage Track2p result failures into registration, ranking, solver, or scoring fixes",
    ),
    (
        "edge-ranking",
        "bayescatrack.experiments.track2p_edge_ranking",
        "Rank manual-GT Track2p edges within pairwise cost/feature matrices",
    ),
    (
        "select-edge-ranking-features",
        "bayescatrack.experiments.edge_ranking_feature_selection",
        "Select calibrated feature names from edge-ranking summary CSVs",
    ),
    (
        "select-structured-objective",
        "bayescatrack.experiments.structured_objective_tuning",
        "Rank benchmark variants by complete-track or other structured metrics",
    ),
    (
        "registration-qa",
        "bayescatrack.experiments.registration_qa_report",
        "Report registration quality on manual-GT Track2p links",
    ),
    (
        "oracle-affine-qa",
        "bayescatrack.experiments.oracle_affine_registration_qa",
        "Compare baseline registration to manual-GT oracle affine geometry",
    ),
    (
        "growth-registration-qa",
        "bayescatrack.experiments.growth_registration_qa",
        "Report spatially resolved growth/deformation registration QA",
    ),
    (
        "validate-track2p-inputs",
        "bayescatrack.experiments.track2p_input_validator",
        "Validate manual-GT ROI coverage before Track2p benchmarks",
    ),
    (
        "audit-manual-gt-rois",
        "bayescatrack.experiments.track2p_roi_index_audit",
        "Audit manual-GT ROI index spaces before Track2p benchmarks",
    ),
    (
        "compare",
        "bayescatrack.experiments.benchmark_comparison",
        "Aggregate benchmark CSVs into a comparison table",
    ),
    (
        "suite",
        "bayescatrack.experiments.benchmark_manifest",
        "Run a JSON benchmark manifest",
    ),
    (
        "validate-suite",
        "bayescatrack.experiments.benchmark_manifest_plan",
        "Validate a JSON benchmark manifest without running benchmarks",
    ),
    (
        "resolve-suite",
        "bayescatrack.experiments.benchmark_manifest_resolver",
        "Resolve benchmark manifest root placeholders into an executable copy",
    ),
)

_BENCHMARK_COMMANDS: dict[str, _BenchmarkCommand] = {
    name: _BenchmarkCommand(module, help_text)
    for name, module, help_text in _BENCHMARK_COMMAND_DATA
}

_BENCHMARK_ALIASES: dict[str, str] = {
    "track2p-component-cleanup": "track2p-policy-component-audit",
    "track2p-component-residual-audit": ("track2p-policy-component-residual-audit"),
    "track2p-residual-audit": "track2p-policy-component-residual-audit",
    "track2p-seed-sensitivity-audit": "track2p-policy-seed-sensitivity-audit",
    "track2p-component-residual-whatif": ("track2p-policy-component-residual-whatif"),
    "track2p-residual-whatif": "track2p-policy-component-residual-whatif",
    "track2p-component-seed-sensitivity-audit": (
        "track2p-policy-seed-sensitivity-audit"
    ),
    "track2p-fragment-stitch-whatif": "track2p-policy-fragment-stitch-whatif",
    "track2p-component-fragment-stitch-whatif": (
        "track2p-policy-fragment-stitch-whatif"
    ),
    "track2p-suffix-stitch-ranking-audit": (
        "track2p-policy-suffix-stitch-ranking-audit"
    ),
    "track2p-component-suffix-stitch-ranking-audit": (
        "track2p-policy-suffix-stitch-ranking-audit"
    ),
    "track2p-coherence-suffix-stitch-whatif": (
        "track2p-policy-coherence-suffix-stitch-whatif"
    ),
    "track2p-component-coherence-suffix-stitch-whatif": (
        "track2p-policy-coherence-suffix-stitch-whatif"
    ),
    "track2p-coherence-suffix-stitch": ("track2p-policy-coherence-suffix-stitch"),
    "track2p-component-coherence-suffix-stitch": (
        "track2p-policy-coherence-suffix-stitch"
    ),
    "track2p-coherence-suffix-teacher-rescue": (
        "track2p-policy-coherence-suffix-teacher-rescue"
    ),
    "track2p-component-coherence-suffix-teacher-rescue": (
        "track2p-policy-coherence-suffix-teacher-rescue"
    ),
    "track2p-coherence-teacher-overlay-audit": (
        "track2p-policy-coherence-teacher-overlay-audit"
    ),
    "track2p-component-coherence-teacher-overlay-audit": (
        "track2p-policy-coherence-teacher-overlay-audit"
    ),
    "track2p-growth-field-residual-audit": (
        "track2p-policy-growth-field-residual-audit"
    ),
    "track2p-component-growth-field-residual-audit": (
        "track2p-policy-growth-field-residual-audit"
    ),
    "track2p-growth-veto-whatif": ("track2p-policy-growth-veto-whatif"),
    "track2p-component-growth-veto-whatif": ("track2p-policy-growth-veto-whatif"),
    "track2p-growth-veto-cleanup": "track2p-policy-growth-veto-cleanup",
    "track2p-component-growth-veto-cleanup": ("track2p-policy-growth-veto-cleanup"),
    "track2p-coherence-suffix-growth-veto-cleanup": (
        "track2p-policy-coherence-suffix-growth-veto-cleanup"
    ),
    "track2p-component-coherence-suffix-growth-veto-cleanup": (
        "track2p-policy-coherence-suffix-growth-veto-cleanup"
    ),
    "track2p-pyrecest-residual-mht-cleanup": (
        "track2p-policy-pyrecest-residual-mht-cleanup"
    ),
    "track2p-component-pyrecest-residual-mht-cleanup": (
        "track2p-policy-pyrecest-residual-mht-cleanup"
    ),
    "track2p-pyrecest-frontier-mht-cleanup": (
        "track2p-policy-pyrecest-frontier-mht-cleanup"
    ),
    "track2p-component-pyrecest-frontier-mht-cleanup": (
        "track2p-policy-pyrecest-frontier-mht-cleanup"
    ),
    "track2p-coherence-suffix-exposure-audit": (
        "track2p-policy-coherence-suffix-exposure-audit"
    ),
    "track2p-component-coherence-suffix-exposure-audit": (
        "track2p-policy-coherence-suffix-exposure-audit"
    ),
    "track2p-coherence-pareto-whatif": ("track2p-policy-coherence-pareto-whatif"),
    "track2p-component-coherence-pareto-whatif": (
        "track2p-policy-coherence-pareto-whatif"
    ),
    "track2p-growth-regularized-assignment": (
        "track2p-policy-growth-regularized-assignment"
    ),
    "track2p-component-growth-regularized-assignment": (
        "track2p-policy-growth-regularized-assignment"
    ),
    "track2p-teacher-free-adjacent-rescue-ranking-audit": (
        "track2p-policy-teacher-free-adjacent-rescue-ranking-audit"
    ),
    "track2p-component-teacher-free-adjacent-rescue-ranking-audit": (
        "track2p-policy-teacher-free-adjacent-rescue-ranking-audit"
    ),
    "track2p-teacher-fn-audit": "track2p-policy-teacher-fn-audit",
    "track2p-component-teacher-fn-audit": "track2p-policy-teacher-fn-audit",
    "track2p-teacher-adjacent-rescue": ("track2p-policy-teacher-adjacent-rescue"),
    "track2p-component-teacher-adjacent-rescue": (
        "track2p-policy-teacher-adjacent-rescue"
    ),
    "track2p-teacher-veto-cleanup": "track2p-policy-teacher-veto-cleanup",
    "track2p-component-teacher-veto-cleanup": ("track2p-policy-teacher-veto-cleanup"),
    "track2p-component-cleanup-sweep": "track2p-policy-component-sweep",
    "track2p-stability-cleanup": "track2p-policy-stability-cleanup",
    "track2p-multisplit-cleanup": "track2p-policy-multisplit-cleanup",
    "track2p-component-multisplit-cleanup": "track2p-policy-multisplit-cleanup",
    "track2p-consensus-cleanup": "track2p-policy-consensus-cleanup",
    "track2p-component-consensus-cleanup": "track2p-policy-consensus-cleanup",
    "track2p-gap-pruned": "track2p-policy-gap-pruned",
    "track2p-gap-rescue-pruned": "track2p-policy-gap-pruned",
    "track2p-policy-gap-rescue-pruned": "track2p-policy-gap-pruned",
    "track2p-gap-component-cleanup": "track2p-policy-gap-component-cleanup",
    "track2p-gap-rescue-component-cleanup": ("track2p-policy-gap-component-cleanup"),
    "track2p-policy-gap-rescue-component-cleanup": (
        "track2p-policy-gap-component-cleanup"
    ),
    "track2p-strict-gated-gap-cleanup": ("track2p-policy-strict-gated-gap-cleanup"),
    "track2p-component-strict-gated-gap-cleanup": (
        "track2p-policy-strict-gated-gap-cleanup"
    ),
    "track2p-component-strict-gap-cleanup": ("track2p-policy-strict-gated-gap-cleanup"),
    "track2p-confidence-strict-gap-cleanup": (
        "track2p-policy-confidence-ordered-strict-gated-gap-cleanup"
    ),
    "track2p-confidence-ordered-strict-gap-cleanup": (
        "track2p-policy-confidence-ordered-strict-gated-gap-cleanup"
    ),
    "track2p-confidence-ordered-strict-gated-gap-cleanup": (
        "track2p-policy-confidence-ordered-strict-gated-gap-cleanup"
    ),
    "track2p-component-confidence-strict-gap-cleanup": (
        "track2p-policy-confidence-ordered-strict-gated-gap-cleanup"
    ),
    "track2p-policy-confidence-strict-gap-cleanup": (
        "track2p-policy-confidence-ordered-strict-gated-gap-cleanup"
    ),
    "track2p-gap-bridge-cleanup": "track2p-policy-gap-bridge-cleanup",
    "track2p-gap-rescue-bridge-cleanup": "track2p-policy-gap-bridge-cleanup",
    "track2p-policy-gap-rescue-bridge-cleanup": ("track2p-policy-gap-bridge-cleanup"),
    "track2p-gap-edge-audit": "track2p-policy-gap-edge-audit",
    "track2p-gap-rescue-edge-audit": "track2p-policy-gap-edge-audit",
    "track2p-gap-consensus-cleanup": "track2p-policy-gap-consensus-cleanup",
    "track2p-gap-rescue-consensus-cleanup": "track2p-policy-gap-consensus-cleanup",
    "track2p-component-gap-consensus-cleanup": "track2p-policy-gap-consensus-cleanup",
    "track2p-gap-consensus-sweep": "track2p-policy-gap-consensus-sweep",
    "track2p-gap-rescue-consensus-sweep": "track2p-policy-gap-consensus-sweep",
    "track2p-gap-consensus-guarded-sweep": "track2p-policy-gap-consensus-guarded-sweep",
    "track2p-guarded-gap-consensus-sweep": "track2p-policy-gap-consensus-guarded-sweep",
    "track2p-gap-rescue-consensus-guarded-sweep": (
        "track2p-policy-gap-consensus-guarded-sweep"
    ),
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
        from bayescatrack.experiments.advanced_improvement_workbench import (
            main as _advanced_main,
        )

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
