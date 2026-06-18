from __future__ import annotations

import pytest
from bayescatrack.experiments import (
    track2p_policy_coherence_suffix_stitch_whatif,
    track2p_policy_coherence_suffix_teacher_rescue,
    track2p_policy_coherence_teacher_overlay_audit,
    track2p_policy_growth_field_residual_audit,
    track2p_policy_growth_veto_cleanup,
    track2p_policy_growth_veto_whatif,
    track2p_policy_pyrecest_calibrated_mht_cleanup,
    track2p_policy_pyrecest_residual_mht_cleanup,
    track2p_policy_teacher_free_adjacent_rescue_ranking_audit,
)


class StopAfterConfig(RuntimeError):
    pass


@pytest.mark.parametrize(
    "module",
    [
        track2p_policy_coherence_suffix_stitch_whatif,
        track2p_policy_coherence_suffix_teacher_rescue,
        track2p_policy_coherence_teacher_overlay_audit,
        track2p_policy_growth_field_residual_audit,
        track2p_policy_growth_veto_cleanup,
        track2p_policy_growth_veto_whatif,
        track2p_policy_pyrecest_calibrated_mht_cleanup,
        track2p_policy_pyrecest_residual_mht_cleanup,
        track2p_policy_teacher_free_adjacent_rescue_ranking_audit,
    ],
)
def test_suffix_family_policy_clis_forward_max_gap(module, monkeypatch, tmp_path):
    captured = {}

    def fake_config(**kwargs):
        captured.update(kwargs)
        raise StopAfterConfig

    monkeypatch.setattr(module, "Track2pBenchmarkConfig", fake_config)

    with pytest.raises(StopAfterConfig):
        module.main(
            [
                "--data",
                str(tmp_path),
                "--max-gap",
                "5",
                "--output",
                str(tmp_path / "out.csv"),
            ]
        )

    assert captured["max_gap"] == 5
