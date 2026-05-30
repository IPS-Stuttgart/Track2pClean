import numpy as np
from bayescatrack import cli
from bayescatrack.experiments import track2p_policy_seed_sensitivity_audit as audit


def test_seed_sensitivity_audit_is_registered() -> None:
    canonical = cli._BENCHMARK_ALIASES["track2p-seed-sensitivity-audit"]

    assert canonical == "track2p-policy-seed-sensitivity-audit"
    assert (
        cli._BENCHMARK_ALIASES["track2p-component-seed-sensitivity-audit"] == canonical
    )
    assert cli._BENCHMARK_COMMANDS[canonical].module == (
        "bayescatrack.experiments.track2p_policy_seed_sensitivity_audit"
    )


def test_resolved_seed_sessions_supports_all_and_csv() -> None:
    assert audit._resolved_seed_sessions("all", n_sessions=3) == (0, 1, 2)
    assert audit._resolved_seed_sessions("0,2", n_sessions=3) == (0, 2)


def test_complete_reference_rows_mark_missing_seed_recoverability() -> None:
    rows = [
        {
            "subject": "s",
            "seed_session": 0,
            "reference_track_id": "1,2,3",
            "status": "false_negative",
            "reason_bucket": "missing seed-session ROI",
            "missing_seed_complete_fn": 1,
            "recoverable_under_other_seed": 0,
            "recovered_seed_sessions": "",
        },
        {
            "subject": "s",
            "seed_session": 1,
            "reference_track_id": "1,2,3",
            "status": "true_positive",
            "reason_bucket": "",
            "missing_seed_complete_fn": 0,
            "recoverable_under_other_seed": 0,
            "recovered_seed_sessions": "",
        },
    ]

    annotated = audit._annotate_recoverability(rows)

    assert annotated[0]["recoverable_under_other_seed"] == 1
    assert annotated[0]["recovered_seed_sessions"] == "1"
    assert annotated[1]["recoverable_under_other_seed"] == 0


def test_complete_reference_track_rows_use_residual_reason() -> None:
    predicted = np.asarray([[1, 2, 9]], dtype=int)
    reference = np.asarray([[1, 2, 3]], dtype=int)
    residual_rows = [
        {
            "error_type": "complete_fn",
            "track_id_or_edge": "1,2,3",
            "occurrence_index": 0,
            "reason_bucket": "fragmented GT track",
        }
    ]

    rows = audit._complete_reference_track_rows(
        subject="s",
        seed_session=0,
        predicted=predicted,
        reference=reference,
        residual_rows=residual_rows,
    )

    assert rows[0]["status"] == "false_negative"
    assert rows[0]["reason_bucket"] == "fragmented GT track"


def test_aggregate_seed_rows_recomputes_micro_f1() -> None:
    rows = [
        {
            "subject": "a",
            "seed_session": 0,
            "pairwise_true_positives": 2,
            "pairwise_false_positives": 0,
            "pairwise_false_negatives": 0,
            "complete_track_true_positives": 1,
            "complete_track_false_positives": 0,
            "complete_track_false_negatives": 0,
            "missing_seed_complete_fns": 0,
            "complete_fns_recovered_under_other_seed": 0,
            "missing_seed_complete_fns_recovered_under_other_seed": 0,
            "reference_complete_tracks": 1,
            "evaluated_prediction_tracks": 1,
            "threshold_method": "min",
            "iou_distance_threshold": 12.0,
            "cell_probability_threshold": 0.5,
            "transform_type": "affine",
        },
        {
            "subject": "b",
            "seed_session": 0,
            "pairwise_true_positives": 0,
            "pairwise_false_positives": 1,
            "pairwise_false_negatives": 1,
            "complete_track_true_positives": 0,
            "complete_track_false_positives": 1,
            "complete_track_false_negatives": 1,
            "missing_seed_complete_fns": 1,
            "complete_fns_recovered_under_other_seed": 1,
            "missing_seed_complete_fns_recovered_under_other_seed": 1,
            "reference_complete_tracks": 1,
            "evaluated_prediction_tracks": 1,
            "threshold_method": "min",
            "iou_distance_threshold": 12.0,
            "cell_probability_threshold": 0.5,
            "transform_type": "affine",
        },
    ]

    aggregate = audit._aggregate_seed_rows(rows)

    assert aggregate[0]["subject"] == "ALL"
    assert aggregate[0]["pairwise_f1_micro"] == 4 / 6
    assert aggregate[0]["complete_track_f1_micro"] == 2 / 4
    assert aggregate[0]["missing_seed_complete_fns"] == 1
