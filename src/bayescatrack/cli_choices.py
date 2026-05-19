"""Shared BayesCaTrack CLI choice helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

T = TypeVar("T")

REGISTRATION_TRANSFORM_EXTRA_CHOICES = (
    "fov-affine",
    "bspline",
    "b-spline",
    "thin-plate-spline",
    "tps",
    "landmark-tps",
    "local-affine-grid",
    "optical-flow",
    "association-guided-bspline",
    "association-guided-b-spline",
    "association-guided-thin-plate-spline",
    "association-guided-tps",
    "association-guided-landmark-tps",
    "association-guided-local-affine-grid",
    "association-guided-optical-flow",
)


def expanded_choices(
    choices: Iterable[T] | None,
    extra_choices: Iterable[T],
) -> tuple[T, ...] | None:
    """Return ``choices`` with ``extra_choices`` inserted before ``"none"``."""

    if choices is None:
        return None
    choices_tuple = tuple(choices)
    if not choices_tuple:
        return choices_tuple

    expanded: list[T] = []
    for value in choices_tuple:
        if value == "none":
            expanded.extend(
                extra_choice
                for extra_choice in extra_choices
                if extra_choice not in expanded
            )
        expanded.append(value)
    if "none" not in choices_tuple:
        expanded.extend(
            extra_choice for extra_choice in extra_choices if extra_choice not in expanded
        )
    return tuple(dict.fromkeys(expanded))


def registration_transform_choices(choices: Iterable[str]) -> tuple[str, ...]:
    """Return transform choices including BayesCaTrack growth-aware aliases."""

    return expanded_choices(choices, REGISTRATION_TRANSFORM_EXTRA_CHOICES) or ()
