from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.association.calibrated_costs import ReferencePairwiseExamples
from bayescatrack.experiments.track2p_teacher_distillation import (
    TeacherDistillationOptions,
    collect_weighted_candidate_examples,
    combine_sample_weights,
    validate_teacher_distillation_options,
)


def test_combine_sample_weights_multiplies_class_and_source_weights():
    combined = combine_sample_weights(
        np.array([2.0, 4.0, 6.0]), np.array([0.5, 0.25, 1.0])
    )

    np.testing.assert_allclose(combined, np.array([1.0, 1.0, 6.0]))


def test_combine_sample_weights_returns_none_for_unweighted_sources():
    assert combine_sample_weights(None, np.ones((3,), dtype=float)) is None


def test_combine_sample_weights_broadcasts_scalar_base_weight():
    combined = combine_sample_weights(2.0, np.array([0.5, 0.25]))

    np.testing.assert_allclose(combined, np.array([1.0, 0.5]))


def test_validate_teacher_distillation_options_rejects_missing_sources():
    with pytest.raises(ValueError, match="At least one"):
        validate_teacher_distillation_options(
            TeacherDistillationOptions(
                include_manual_training_labels=False,
                include_teacher_training_labels=False,
            )
        )


def test_validate_teacher_distillation_options_rejects_nonpositive_weights():
    with pytest.raises(ValueError, match="teacher_label_weight"):
        validate_teacher_distillation_options(
            TeacherDistillationOptions(teacher_label_weight=0.0)
        )


def test_collect_weighted_candidate_examples_attaches_source_weight():
    blocks = (
        ReferencePairwiseExamples(
            session_a=0,
            session_b=1,
            features=np.array([[[0.0], [1.0]], [[2.0], [3.0]]], dtype=float),
            labels=np.array([[1, 0], [0, 1]], dtype=int),
            reference_roi_indices=np.array([0, 1], dtype=int),
            measurement_roi_indices=np.array([0, 1], dtype=int),
            feature_names=("dummy",),
        ),
    )

    examples = collect_weighted_candidate_examples(blocks, source_weight=0.25)

    assert examples.features.shape[-1] == 1
    assert examples.labels.ndim == 1
    assert examples.source_weights is not None
    assert examples.source_weights.shape == examples.labels.shape
    np.testing.assert_allclose(
        examples.source_weights, np.full(examples.labels.shape, 0.25, dtype=float)
    )
