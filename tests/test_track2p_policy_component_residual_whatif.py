from __future__ import annotations

import csv

from bayescatrack import cli
from bayescatrack.experiments import track2p_policy_component_residual_whatif as whatif


def test_component_residual_whatif_is_registered() -> None:
    canonical = cli._BENCHMARK_ALIASES["track2p-component-residual-whatif"]

    assert canonical == "track2p-policy-component-residual-whatif"
    assert cli._BENCHMARK_ALIASES["track2p-residual-whatif"] == canonical
    assert cli._BENCHMARK_COMMANDS[canonical].module == (
        "bayescatrack.experiments.track2p_policy_component_residual_whatif"
    )


def test_residual_whatif_scores_pairwise_fn_and_fp_edits() -> None:
    base = whatif.MicroCounts(
        pairwise_tp=586,
        pairwise_fp=26,
        pairwise_fn=19,
        complete_tp=56,
        complete_fp=3,
        complete_fn=4,
    )
    rows = whatif.residual_whatif_rows(
        [
            {
                "subject": "jm046",
                "error_type": "pairwise_fn",
                "track_id_or_edge": "1:10->2:11",
                "reason_bucket": "Track2p-supported missed adjacent edge",
                "is_track2p_supported": "1",
                "is_policy_supported": "0",
                "is_gap_rescue_supported": "0",
                "is_component_cleanup_affected": "0",
            },
            {
                "subject": "jm038",
                "error_type": "pairwise_fp",
                "track_id_or_edge": "3:12->4:13",
                "reason_bucket": "Bayes-only false continuation",
                "is_track2p_supported": "0",
                "is_policy_supported": "1",
                "is_gap_rescue_supported": "0",
                "is_component_cleanup_affected": "1",
            },
        ],
        base,
    )

    fn_row = next(row for row in rows if row["edit_type"] == "add_pairwise_fn_as_tp")
    assert fn_row["support_bucket"] == "track2p"
    assert fn_row["new_pairwise_tp"] == 587
    assert fn_row["new_pairwise_fp"] == 26
    assert fn_row["new_pairwise_fn"] == 18
    assert fn_row["pairwise_f1_delta"] > 0

    fp_row = next(row for row in rows if row["edit_type"] == "remove_pairwise_fp")
    assert fp_row["support_bucket"] == "policy+cleanup-affected"
    assert fp_row["new_pairwise_tp"] == 586
    assert fp_row["new_pairwise_fp"] == 25
    assert fp_row["new_pairwise_fn"] == 19
    assert fp_row["pairwise_f1_delta"] > 0


def test_residual_whatif_bundle_rows_accumulate_same_support_family() -> None:
    base = whatif.MicroCounts(
        pairwise_tp=586,
        pairwise_fp=26,
        pairwise_fn=19,
        complete_tp=56,
        complete_fp=3,
        complete_fn=4,
    )
    candidates = whatif.residual_whatif_rows(
        [
            {
                "subject": "jm046",
                "error_type": "pairwise_fn",
                "track_id_or_edge": "1:10->2:11",
                "reason_bucket": "Track2p-supported missed adjacent edge",
                "is_track2p_supported": "1",
                "is_policy_supported": "0",
                "is_gap_rescue_supported": "0",
                "is_component_cleanup_affected": "0",
            },
            {
                "subject": "jm046",
                "error_type": "pairwise_fn",
                "track_id_or_edge": "2:11->3:12",
                "reason_bucket": "Track2p-supported missed adjacent edge",
                "is_track2p_supported": "1",
                "is_policy_supported": "0",
                "is_gap_rescue_supported": "0",
                "is_component_cleanup_affected": "0",
            },
            {
                "subject": "jm038",
                "error_type": "pairwise_fp",
                "track_id_or_edge": "3:12->4:13",
                "reason_bucket": "Bayes-only false continuation",
                "is_track2p_supported": "0",
                "is_policy_supported": "1",
                "is_gap_rescue_supported": "0",
                "is_component_cleanup_affected": "1",
            },
        ],
        base,
    )

    bundles = whatif.residual_whatif_bundle_rows(candidates, base, max_bundle_size=2)

    bundle = next(
        row
        for row in bundles
        if row["edit_type"] == "add_pairwise_fn_as_tp"
        and row["support_bucket"] == "track2p"
        and row["bundle_size"] == 2
    )
    single_fn = next(
        row for row in candidates if row["edit_type"] == "add_pairwise_fn_as_tp"
    )
    assert bundle["new_pairwise_tp"] == 588
    assert bundle["new_pairwise_fp"] == 26
    assert bundle["new_pairwise_fn"] == 17
    assert bundle["pairwise_tp_delta"] == 2
    assert bundle["pairwise_fn_delta"] == -2
    assert bundle["pairwise_f1_delta"] > single_fn["pairwise_f1_delta"]


def test_load_base_counts_sums_subject_rows(tmp_path) -> None:
    path = tmp_path / "component_cleanup.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "subject",
                "pairwise_true_positives",
                "pairwise_false_positives",
                "pairwise_false_negatives",
                "complete_track_true_positives",
                "complete_track_false_positives",
                "complete_track_false_negatives",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "subject": "a",
                "pairwise_true_positives": 2,
                "pairwise_false_positives": 1,
                "pairwise_false_negatives": 3,
                "complete_track_true_positives": 4,
                "complete_track_false_positives": 0,
                "complete_track_false_negatives": 1,
            }
        )
        writer.writerow(
            {
                "subject": "b",
                "pairwise_true_positives": 5,
                "pairwise_false_positives": 0,
                "pairwise_false_negatives": 1,
                "complete_track_true_positives": 6,
                "complete_track_false_positives": 2,
                "complete_track_false_negatives": 0,
            }
        )

    counts = whatif.load_base_counts(path)

    assert counts == whatif.MicroCounts(
        pairwise_tp=7,
        pairwise_fp=1,
        pairwise_fn=4,
        complete_tp=10,
        complete_fp=2,
        complete_fn=1,
    )
