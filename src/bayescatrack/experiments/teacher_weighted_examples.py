"""Weak-teacher weighting utilities for Track2p/Bayes/manual edge audits."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TeacherWeightConfig:
    """Sample weights for manual GT and weak Track2p teacher labels."""

    manual_positive_weight: float = 1.0
    manual_negative_weight: float = 1.0
    teacher_positive_weight: float = 0.25
    teacher_negative_weight: float = 0.10
    disagreement_downweight: float = 0.25

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:  # type: ignore[attr-defined]
            value = float(getattr(self, name))
            if value < 0.0 or not np.isfinite(value):
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)


def weighted_teacher_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: TeacherWeightConfig | None = None,
) -> list[dict[str, Any]]:
    """Attach fold-internal weak-supervision labels and weights to audit rows.

    Manual GT remains the high-trust label.  Track2p-only edges can be used as
    low-weight weak positives; Bayes-only/Track2p-negative edges become weak
    negatives.  This helper intentionally does not alter benchmark evaluation.
    """

    cfg = config or TeacherWeightConfig()
    out: list[dict[str, Any]] = []
    for row in rows:
        manual = _truthy(row.get("in_ground_truth"))
        teacher = _truthy(row.get("in_track2p"))
        bayes = _truthy(row.get("in_bayes"))
        if manual:
            label = 1
            weight = cfg.manual_positive_weight
            source = "manual_gt"
        elif teacher:
            label = 1
            weight = cfg.teacher_positive_weight
            source = "track2p_weak_teacher"
        elif bayes:
            label = 0
            weight = cfg.teacher_negative_weight
            source = "bayes_false_positive_candidate"
        else:
            label = 0
            weight = cfg.manual_negative_weight
            source = "unmatched_negative"
        if manual != teacher:
            weight *= cfg.disagreement_downweight
        out.append(
            {
                **dict(row),
                "training_label": int(label),
                "training_weight": float(weight),
                "training_label_source": source,
                "manual_teacher_disagreement": int(manual != teacher),
            }
        )
    return out


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


__all__ = ("TeacherWeightConfig", "weighted_teacher_rows")
