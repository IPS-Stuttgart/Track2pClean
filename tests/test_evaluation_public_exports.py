from __future__ import annotations

import bayescatrack.evaluation as evaluation
from bayescatrack.evaluation import track2p_metrics


def test_evaluation_facade_imports_and_exports_track_scoring_symbols():
    assert "score_track_matrices" in evaluation.__all__
    assert "normalize_track_matrix" in evaluation.__all__
    assert evaluation.normalize_track_matrix is track2p_metrics.normalize_track_matrix


def test_track2p_metrics_exports_track_scoring_facade_symbols():
    assert "score_track_matrices" in track2p_metrics.__all__
    assert "normalize_track_matrix" in track2p_metrics.__all__
    assert "score_track_matrix_against_reference" in track2p_metrics.__all__
