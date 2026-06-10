from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.experiments import (
    track2p_policy_pyrecest_residual_mht_cleanup as residual_mht,
)
from bayescatrack.experiments.track2p_benchmark import (
    GROUND_TRUTH_REFERENCE_SOURCE,
    SubjectBenchmarkResult,
)


def _result() -> residual_mht.PyRecEstResidualMHTResult:
    return residual_mht.PyRecEstResidualMHTResult(
        results=(
            SubjectBenchmarkResult(
                subject="jm046",
                variant="test",
                method="global-assignment",
                scores={"pairwise_f1_micro": 1.0},
                n_sessions=7,
                reference_source=GROUND_TRUTH_REFERENCE_SOURCE,
            ),
        ),
        candidate_rows=(
            {
                "subject": "jm046",
                "pyrecest_candidate_id": "jm046:5:6:2309:1210:0",
                "pyrecest_candidate": 1,
                "selected_by_pyrecest_mht": 1,
                "applied_by_pyrecest_mht": 1,
            },
        ),
        summary_rows=(),
    )


def _required_args(tmp_path) -> list[str]:
    return [
        "--data",
        str(tmp_path),
        "--reference",
        str(tmp_path),
        "--reference-kind",
        "manual-gt",
        "--output",
        str(tmp_path / "scores.csv"),
    ]


def test_main_writes_candidate_output(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        residual_mht,
        "run_track2p_policy_pyrecest_residual_mht_cleanup",
        lambda *args, **kwargs: _result(),
    )
    candidates = tmp_path / "candidates.csv"

    exit_status = residual_mht.main(
        [
            *_required_args(tmp_path),
            "--candidate-output",
            str(candidates),
            "--format",
            "csv",
        ]
    )

    assert exit_status == 0
    assert (tmp_path / "scores.csv").exists()
    assert candidates.exists()
    assert "jm046:5:6:2309:1210:0" in candidates.read_text(encoding="utf-8")


def test_main_rejects_teacher_rescue_base(tmp_path) -> None:
    with pytest.raises(SystemExit):
        residual_mht.main(
            [
                *_required_args(tmp_path),
                "--growth-veto-base",
                "teacher-rescue",
            ]
        )


def test_explicit_legacy_veto_cap_seeds_mht_caps(monkeypatch, tmp_path) -> None:
    captured = {}
    monkeypatch.setattr(
        residual_mht,
        "run_track2p_policy_pyrecest_residual_mht_cleanup",
        lambda *args, **kwargs: captured.update(kwargs) or _result(),
    )

    exit_status = residual_mht.main(
        [
            *_required_args(tmp_path),
            "--max-vetoes-per-subject",
            "1",
        ]
    )

    assert exit_status == 0
    mht_options = captured["mht_options"]
    assert mht_options.candidate_top_k == 1
    assert mht_options.max_edits_per_subject == 1


def test_mht_specific_caps_override_legacy_veto_cap(monkeypatch, tmp_path) -> None:
    captured = {}
    monkeypatch.setattr(
        residual_mht,
        "run_track2p_policy_pyrecest_residual_mht_cleanup",
        lambda *args, **kwargs: captured.update(kwargs) or _result(),
    )

    exit_status = residual_mht.main(
        [
            *_required_args(tmp_path),
            "--max-vetoes-per-subject",
            "1",
            "--mht-candidate-top-k",
            "3",
            "--mht-max-edits-per-subject",
            "2",
        ]
    )

    assert exit_status == 0
    mht_options = captured["mht_options"]
    assert mht_options.candidate_top_k == 3
    assert mht_options.max_edits_per_subject == 2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"candidate_top_k": 0}, "candidate_top_k must be a positive integer"),
        ({"candidate_top_k": 1.5}, "candidate_top_k must be a positive integer"),
        ({"candidate_top_k": True}, "candidate_top_k must be finite"),
        (
            {"max_edits_per_subject": -1},
            "max_edits_per_subject must be a non-negative integer",
        ),
        ({"max_hypotheses": np.inf}, "max_hypotheses must be finite"),
        ({"edit_penalty": -0.1}, "edit_penalty must be finite and non-negative"),
        ({"score_threshold": np.nan}, "score_threshold must be finite"),
    ],
)
def test_mht_options_reject_silent_candidate_knob_coercions(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        residual_mht.PyRecEstResidualMHTOptions(**kwargs)
