"""Paper-facing bundle decision for FullMHT identity-history evidence.

This helper deliberately sits above the individual decision helpers.  It prevents
cherry-picking an optional scan-pruning or terminal-completion add-on when the
central identity-history row has not passed its own manifest, sensitivity,
subject-support, and label-free exposure gates.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CORE_ROW = "FullMHTIdentityHistory"
CORE_MHT_RESULT = "identity_complete_history_advantage"
CORE_SENSITIVITY_RESULT = "stable_plateau"
CORE_EXPOSURE_RESULT = "bounded_exposure"
CORE_SUBJECT_SUPPORT_RESULT = "stable_subject_support"


def load_decision(path: Path) -> dict[str, Any]:
    """Load a JSON decision artifact."""

    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Decision artifact {path} must contain a JSON object")
    return data


def evaluate_identity_history_bundle(
    identity_promotion: Mapping[str, Any],
    *,
    subject_support: Mapping[str, Any] | None = None,
    scan_pruning_promotion: Mapping[str, Any] | None = None,
    terminal_completion: Mapping[str, Any] | None = None,
    local_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine central and optional FullMHT identity-history evidence."""

    core_status = str(identity_promotion.get("status", "incomplete"))
    core_mht_result = str(identity_promotion.get("mht_vs_local_result", "incomplete"))
    core_sensitivity_result = str(identity_promotion.get("sensitivity_result", "incomplete"))
    core_exposure_result = str(identity_promotion.get("exposure_result", "incomplete"))
    subject = _subject_support_block(subject_support)
    subject_support_ok = subject["result"] == CORE_SUBJECT_SUPPORT_RESULT
    core_evidence_ok = (
        core_mht_result == CORE_MHT_RESULT
        and core_sensitivity_result == CORE_SENSITIVITY_RESULT
        and core_exposure_result == CORE_EXPOSURE_RESULT
        and subject_support_ok
    )
    core_evidence_result = "complete_core_evidence" if core_evidence_ok else "inconsistent_core_evidence"
    scan = _scan_pruning_block(scan_pruning_promotion)
    terminal = _terminal_completion_block(terminal_completion)
    local = _local_context_block(local_context)

    if core_status == "promotable_after_review" and core_evidence_ok:
        status = "promotable_core_method"
        paper_row = CORE_ROW
        recommendation = "promote the central identity-history row; treat add-ons as separate variants"
    elif core_status == "incomplete" or (
        core_status == "promotable_after_review"
        and subject["status"] in {"not_evaluated", "incomplete"}
    ):
        status = "incomplete"
        paper_row = ""
        recommendation = "rerun the central identity-history promotion and subject-support gates before interpreting add-ons"
    else:
        status = "not_promotable_core_method"
        paper_row = ""
        if core_status == "promotable_after_review":
            recommendation = "rerun or inspect central promotion gates; status and evidence fields disagree"
        else:
            recommendation = "do not promote optional add-ons because the central identity-history row failed"

    optional_variants: list[str] = []
    exploratory_variants: list[str] = []
    if status == "promotable_core_method":
        if scan["status"] == "candidate_addon":
            optional_variants.append("IdentityHistoryScanPruning")
        elif scan["status"] not in {"not_evaluated", "incomplete"}:
            exploratory_variants.append("IdentityHistoryScanPruning")
        if terminal["status"] == "candidate_addon":
            optional_variants.append("FullMHTIdentityHistoryCompletion")
        elif terminal["status"] not in {"not_evaluated", "incomplete"}:
            exploratory_variants.append("FullMHTIdentityHistoryCompletion")

    return {
        "status": status,
        "paper_row": paper_row,
        "recommendation": recommendation,
        "core_status": core_status,
        "core_evidence_result": core_evidence_result,
        "core_mht_vs_local_result": core_mht_result,
        "core_sensitivity_result": core_sensitivity_result,
        "core_exposure_result": core_exposure_result,
        "subject_support": subject,
        "scan_pruning": scan,
        "terminal_completion": terminal,
        "local_context": local,
        "optional_variants": optional_variants,
        "exploratory_variants": exploratory_variants,
        "guardrail": (
            "optional add-ons are ignored for promotion unless the central "
            "FullMHTIdentityHistory gate is promotable_after_review with complete, "
            "consistent core and subject-support evidence"
        ),
    }


def format_bundle_markdown(decision: Mapping[str, Any]) -> str:
    """Format a compact paper-facing bundle decision note."""

    subject = dict(decision.get("subject_support", {}))
    scan = dict(decision.get("scan_pruning", {}))
    terminal = dict(decision.get("terminal_completion", {}))
    local = dict(decision.get("local_context", {}))
    lines = [
        "# FullMHT Identity-History Bundle Decision",
        "",
        f"Status: `{decision.get('status', '')}`",
        f"Paper row: `{decision.get('paper_row', '')}`",
        f"Core gate: `{decision.get('core_status', '')}`",
        f"Core evidence: `{decision.get('core_evidence_result', '')}`",
        f"MHT-vs-local: `{decision.get('core_mht_vs_local_result', '')}`",
        f"Sensitivity: `{decision.get('core_sensitivity_result', '')}`",
        f"Exposure: `{decision.get('core_exposure_result', '')}`",
        f"Subject support: `{subject.get('result', '')}`",
        f"Recommendation: {decision.get('recommendation', '')}",
        "",
        "| optional evidence | status | result |",
        "| --- | --- | --- |",
        "| scan pruning | {status} | {result} |".format(
            status=scan.get("status", ""),
            result=scan.get("result", ""),
        ),
        "| terminal completion | {status} | {result} |".format(
            status=terminal.get("status", ""),
            result=terminal.get("result", ""),
        ),
        "| local context | {status} | {result} |".format(
            status=local.get("status", ""),
            result=local.get("result", ""),
        ),
        "",
        "Optional variants: {variants}".format(
            variants=", ".join(str(item) for item in decision.get("optional_variants", ()))
            or "none"
        ),
        "Exploratory variants: {variants}".format(
            variants=", ".join(str(item) for item in decision.get("exploratory_variants", ()))
            or "none"
        ),
        f"Guardrail: {decision.get('guardrail', '')}",
    ]
    return "\n".join(lines)


def write_bundle_decision(
    decision: Mapping[str, Any], output: Path, *, output_format: str
) -> None:
    """Write the bundle decision as Markdown or JSON."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output.write_text(json.dumps(dict(decision), indent=2) + "\n", encoding="utf-8")
        return
    output.write_text(format_bundle_markdown(decision) + "\n", encoding="utf-8")


def _subject_support_block(decision: Mapping[str, Any] | None) -> dict[str, str]:
    if decision is None:
        return {"status": "not_evaluated", "result": "not_evaluated"}
    status = str(decision.get("status", "incomplete"))
    result = str(decision.get("subject_support_result", "incomplete"))
    if status != "complete":
        block_status = "incomplete"
    elif result == CORE_SUBJECT_SUPPORT_RESULT:
        block_status = "supporting_component"
    else:
        block_status = "exploratory"
    return {
        "status": block_status,
        "result": result,
        "gate_status": status,
    }


def _scan_pruning_block(decision: Mapping[str, Any] | None) -> dict[str, str]:
    if decision is None:
        return {"status": "not_evaluated", "result": "not_evaluated"}
    status = str(decision.get("status", "incomplete"))
    result = str(decision.get("benchmark_result", "incomplete"))
    exposure = str(decision.get("exposure_result", "incomplete"))
    if status == "promotable_after_review":
        block_status = "candidate_addon"
    elif status == "incomplete":
        block_status = "incomplete"
    else:
        block_status = "exploratory"
    return {
        "status": block_status,
        "result": result,
        "exposure_result": exposure,
        "gate_status": status,
    }


def _terminal_completion_block(decision: Mapping[str, Any] | None) -> dict[str, str]:
    if decision is None:
        return {"status": "not_evaluated", "result": "not_evaluated"}
    status = str(decision.get("status", "incomplete"))
    result = str(decision.get("terminal_completion_result", "incomplete"))
    if status != "complete":
        block_status = "incomplete"
    elif result == "terminal_completion_stable_gain":
        block_status = "candidate_addon"
    else:
        block_status = "exploratory"
    return {
        "status": block_status,
        "result": result,
        "gate_status": status,
    }


def _local_context_block(decision: Mapping[str, Any] | None) -> dict[str, str]:
    if decision is None:
        return {"status": "not_evaluated", "result": "not_evaluated"}
    status = str(decision.get("status", "incomplete"))
    result = str(decision.get("local_context_result", "incomplete"))
    if status != "complete":
        block_status = "incomplete"
    elif result == "history_dynamics_stable_gain":
        block_status = "supporting_component"
    else:
        block_status = "exploratory"
    return {
        "status": block_status,
        "result": result,
        "gate_status": status,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bayescatrack.experiments.full_mht_identity_history_bundle_decision",
        description="Combine FullMHT identity-history promotion and optional add-on decisions.",
    )
    parser.add_argument("identity_promotion_json", type=Path)
    parser.add_argument("--subject-support-json", type=Path, default=None)
    parser.add_argument("--scan-pruning-promotion-json", type=Path, default=None)
    parser.add_argument("--terminal-completion-json", type=Path, default=None)
    parser.add_argument("--local-context-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    decision = evaluate_identity_history_bundle(
        load_decision(args.identity_promotion_json),
        subject_support=(
            None if args.subject_support_json is None else load_decision(args.subject_support_json)
        ),
        scan_pruning_promotion=(
            None
            if args.scan_pruning_promotion_json is None
            else load_decision(args.scan_pruning_promotion_json)
        ),
        terminal_completion=(
            None
            if args.terminal_completion_json is None
            else load_decision(args.terminal_completion_json)
        ),
        local_context=(
            None if args.local_context_json is None else load_decision(args.local_context_json)
        ),
    )
    if args.output is not None:
        write_bundle_decision(decision, args.output, output_format=str(args.format))
    elif args.format == "json":
        print(json.dumps(decision, indent=2))
    else:
        print(format_bundle_markdown(decision))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
