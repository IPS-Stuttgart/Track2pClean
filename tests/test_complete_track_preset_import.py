from __future__ import annotations

from bayescatrack.experiments import track2p_complete_track_preset as preset


def test_defaults_are_complete_track_oriented() -> None:
    assert preset.DEFAULT_OBJECTIVE == "complete_track_f1_micro"
    assert "0.75" in preset.DEFAULT_SPLIT_RISK_THRESHOLDS
    assert "false" in preset.DEFAULT_REQUIRE_COMPLETE_TRACK_OPTIONS
