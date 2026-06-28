"""Evaluation helpers for BayesCaTrack benchmarks."""

from . import _calibration_label_validation as _calibration_label_validation
from . import (
    _edge_ranking_feature_name_validation as _edge_ranking_feature_name_validation,
)
from . import _edge_ranking_roi_validation as _edge_ranking_roi_validation
from . import _track_matrix_vector_validation as _track_matrix_vector_validation
from . import _track_subset_duplicate_validation as _subset_validation
from . import _track_subset_string_validation as _subset_string_validation
from . import calibration_diagnostics as _calibration_diagnostics
from . import complete_track_scores as _scores
from . import track_error_ledger as _track_error_ledger


def _validate_calibration_probability_label_inputs(probabilities, labels):
    return _calibration_label_validation.checked_probability_label_arrays(
        probabilities,
        labels,
        _calibration_diagnostics._as_probability_vector,  # pylint: disable=protected-access
    )


_calibration_diagnostics._validate_probability_label_inputs = (  # pylint: disable=protected-access
    _validate_calibration_probability_label_inputs
)

_SCORE_EXPORTS = (
    "complete_track_set",
    "normalize_track_matrix",
    "pairwise_track_set",
    "reference_fragment_counts",
    "score_complete_tracks",
    "score_false_continuations",
    "score_fragmentation",
    "score_pairwise_tracks",
    "score_track_matrices",
    "summarize_tracks",
    "track_lengths",
)

_edge_ranking_roi_validation.install_edge_ranking_roi_validation()
_edge_ranking_feature_name_validation.install_edge_ranking_feature_name_validation()
_track_matrix_vector_validation.install_track_matrix_vector_input_validation(_scores)
_subset_string_validation.install_track_subset_string_validation(_scores)
_subset_validation.install_track_subset_duplicate_validation(_scores)

# Import facades only after installing score wrappers: track2p_metrics binds
# score_track_matrices during import and must see the patched implementation.
from . import track2p_metrics as _track2p_metrics  # noqa: E402

brier_score = _calibration_diagnostics.brier_score
CalibrationBinRow = _calibration_diagnostics.CalibrationBinRow
calibration_summary = _calibration_diagnostics.calibration_summary
complete_track_set = _scores.complete_track_set
expected_calibration_error = _calibration_diagnostics.expected_calibration_error
format_reliability_bin_table = _calibration_diagnostics.format_reliability_bin_table
maximum_calibration_error = _calibration_diagnostics.maximum_calibration_error
normalize_track_matrix = _track2p_metrics.normalize_track_matrix
pairwise_track_set = _scores.pairwise_track_set
reference_fragment_counts = _scores.reference_fragment_counts
reliability_bin_table = _calibration_diagnostics.reliability_bin_table
score_complete_tracks = _scores.score_complete_tracks
score_false_continuations = _scores.score_false_continuations
score_fragmentation = _scores.score_fragmentation
score_pairwise_tracks = _scores.score_pairwise_tracks
score_track_matrices = _scores.score_track_matrices
summarize_track_errors = _track_error_ledger.summarize_track_errors
summarize_tracks = _scores.summarize_tracks
track_error_ledger = _track_error_ledger.track_error_ledger
track_lengths = _scores.track_lengths
score_track_matrix_against_reference = (
    _track2p_metrics.score_track_matrix_against_reference
)

__all__ = (
    list(_calibration_diagnostics.__all__)
    + list(_SCORE_EXPORTS)
    + list(_track_error_ledger.__all__)
    + [
        "score_track_matrix_against_reference",
    ]
)
