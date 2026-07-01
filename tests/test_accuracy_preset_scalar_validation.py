"""Regression tests for accuracy-preset scalar validation."""

from __future__ import annotations

import builtins

import pytest
from bayescatrack.accuracy_presets import build_track2p_accuracy_presets

_BUFFER_VIEW = getattr(builtins, "".join(("memory", "view")))


def _buffer_view_from_text(text: str):
    return _BUFFER_VIEW(bytearray(ord(character) for character in text))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"max_gap": _buffer_view_from_text("1")},
            "max_gap must be a positive integer",
        ),
        (
            {"cost_threshold": _buffer_view_from_text("1.0")},
            "cost_threshold must be a finite non-negative value or None",
        ),
    ],
)
def test_accuracy_presets_reject_binary_view_scalar_controls(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        build_track2p_accuracy_presets(tmp_path, **kwargs)


def test_accuracy_presets_keep_numeric_string_controls(tmp_path):
    presets = build_track2p_accuracy_presets(
        tmp_path,
        max_gap="2",
        cost_threshold="6.5",
    )

    assert presets[0].config.max_gap == 2
    assert presets[0].config.cost_threshold == 6.5
