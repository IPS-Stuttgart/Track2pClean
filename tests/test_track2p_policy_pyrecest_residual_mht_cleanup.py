from __future__ import annotations

from bayescatrack.experiments import track2p_policy_pyrecest_residual_mht_cleanup as residual_mht
from bayescatrack.experiments.track2p_benchmark import (
    GROUND_TRUTH_REFERENCE_SOURCE,
    SubjectBenchmarkResult,
)


def test_main_writes_candidate_output(monkeypatch, tmp_path) -> None:
    result = residual_mht.PyRecEstResidualMHTResult(
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
    monkeypatch.setattr(
        residual_mht,
        "run_track2p_policy_pyrecest_residual_mht_cleanup",
        lambda *args, **kwargs: result,
    )
    output = tmp_path / "scores.csv"
    candidates = tmp_path / "candidates.csv"

    exit_status = residual_mht.main(
        [
            "--data",
            str(tmp_path),
            "--reference",
            str(tmp_path),
            "--reference-kind",
            "manual-gt",
            "--output",
            str(output),
            "--candidate-output",
            str(candidates),
            "--format",
            "csv",
        ]
    )

    assert exit_status == 0
    assert output.exists()
    assert candidates.exists()
    assert "jm046:5:6:2309:1210:0" in candidates.read_text(encoding="utf-8")
