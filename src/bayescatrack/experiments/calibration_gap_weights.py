"""Sample-weight helpers for gap-aware Track2p calibration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from bayescatrack.experiments.calibration_hard_negatives import (
    balanced_binary_sample_weights,
)


def balanced_binary_gap_sample_weights(
    features: Any,
    labels: Any,
    feature_names: Sequence[str] | str,
    *,
    gap_feature_name: str = "session_gap",
    missing_gap: str = "binary",
) -> np.ndarray:
    """Return inverse-frequency weights balanced jointly by label and session gap.

    Track2p-style calibration pools adjacent-session and skip-session association
    examples. If one gap dominates the candidate set, a plain binary class
    balance can still leave the model under-trained on rarer skip edges. This
    helper equalizes the total training weight of each observed
    ``(binary_label, session_gap)`` group while keeping the overall mean weight
    equal to one.

    Parameters
    ----------
    features:
        Pairwise feature vectors with shape ``(..., n_features)``.
    labels:
        Binary labels with shape matching ``features.shape[:-1]``.
    feature_names:
        Names corresponding to the last feature axis.
    gap_feature_name:
        Feature plane that contains the positive session gap. The calibrated
        default feature set contains ``"session_gap"``.
    missing_gap:
        Policy when ``gap_feature_name`` is absent. ``"binary"`` falls back to
        ordinary binary inverse-frequency weights. ``"raise"`` raises a
        ``ValueError``.
    """

    feature_array = np.asarray(features, dtype=float)
    label_array = np.asarray(labels, dtype=int)
    if feature_array.ndim < 2:
        raise ValueError("features must have shape (..., n_features)")
    if label_array.shape != feature_array.shape[:-1]:
        raise ValueError("labels must match features.shape[:-1]")

    names = _feature_name_tuple(feature_names)
    if feature_array.shape[-1] != len(names):
        raise ValueError("feature_names length must match features.shape[-1]")
    if gap_feature_name not in names:
        if missing_gap == "binary":
            return balanced_binary_sample_weights(label_array).reshape(
                label_array.shape
            )
        if missing_gap == "raise":
            raise ValueError(f"gap feature {gap_feature_name!r} is not present")
        raise ValueError("missing_gap must be either 'binary' or 'raise'")

    flat_labels = label_array.reshape(-1)
    gap_values = feature_array[..., names.index(gap_feature_name)].reshape(-1)
    finite_gap_values = gap_values[np.isfinite(gap_values)]
    if finite_gap_values.size == 0:
        gap_values = np.ones_like(gap_values, dtype=float)
    else:
        fill_value = float(np.median(finite_gap_values))
        gap_values = np.nan_to_num(
            gap_values,
            nan=fill_value,
            posinf=fill_value,
            neginf=fill_value,
        )

    weights = _inverse_group_frequency_weights(flat_labels != 0, gap_values)
    return weights.reshape(label_array.shape)


def _inverse_group_frequency_weights(
    labels: np.ndarray, gaps: np.ndarray
) -> np.ndarray:
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    gaps = np.asarray(gaps, dtype=float).reshape(-1)
    if labels.shape != gaps.shape:
        raise ValueError("labels and gaps must have matching flat shapes")
    n_examples = int(labels.size)
    weights = np.ones(n_examples, dtype=float)
    if n_examples == 0:
        return weights

    rounded_gaps = np.round(gaps, decimals=12)
    group_keys = sorted(
        {(bool(label), float(gap)) for label, gap in zip(labels, rounded_gaps)},
        key=lambda item: (item[1], item[0]),
    )
    if not group_keys:
        return weights

    target_mass = n_examples / float(len(group_keys))
    for label, gap in group_keys:
        mask = (labels == label) & (rounded_gaps == gap)
        group_count = int(np.sum(mask))
        if group_count:
            weights[mask] = target_mass / float(group_count)
    return weights


def _feature_name_tuple(feature_names: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(feature_names, str):
        return (feature_names,)
    return tuple(feature_names)
