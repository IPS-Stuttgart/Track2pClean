from __future__ import annotations

from bayescatrack.experiments import (
    track2p_policy_coherence_suffix_teacher_action_specific as wrapper,
)


def test_default_args_match_focused_action_specific_probe() -> None:
    args = wrapper.default_args()

    assert args == (
        "--teacher-edge-order",
        "dynamic-seed-cell-confidence",
        "--teacher-action-filter",
        "target-extension-or-seed-source-backfill",
        "--teacher-feature-preset",
        "none",
        "--target-extension-feature-preset",
        "moderate-iou-cell-confidence",
        "--seed-source-feature-preset",
        "seed-source-cell-confident",
        "--no-allow-source-backfill",
        "--allow-seed-source-backfill",
        "--allow-completing-seed-source-backfill",
        "--min-teacher-component-observations",
        "2",
        "--max-applied-teacher-edits",
        "2",
    )


def test_main_prepends_default_args(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_main(argv: list[str]) -> int:
        calls.append(argv)
        return 7

    monkeypatch.setattr(wrapper, "_main", fake_main)

    rc = wrapper.main(["--output", "out.csv"])

    assert rc == 7
    assert calls == [[*wrapper.default_args(), "--output", "out.csv"]]
