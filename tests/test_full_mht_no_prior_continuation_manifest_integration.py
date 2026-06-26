from __future__ import annotations

import json
from pathlib import Path

from bayescatrack.experiments import _full_mht_manifest_integration as full_mht_manifest
from bayescatrack.experiments import benchmark_manifest as bm


def test_no_prior_continuation_fields_are_manifest_options() -> None:
    kwargs = bm._runner_kwargs(
        {
            "beam_width": 4,
            "no_prior_continuation_likelihood_weight": 1.25,
            "no_prior_continuation_min_anchor_registered_iou": 0.72,
            "no_prior_continuation_max_anchor_rank": 1,
            "no_prior_continuation_min_examples_per_class": 3,
            "no_prior_continuation_score_clip": 6.0,
        },
        "track2p-policy-full-mht",
    )

    assert kwargs["beam_width"] == 4
    assert kwargs["no_prior_continuation_likelihood_weight"] == 1.25
    assert kwargs["no_prior_continuation_min_anchor_registered_iou"] == 0.72
    assert kwargs["no_prior_continuation_max_anchor_rank"] == 1
    assert kwargs["no_prior_continuation_min_examples_per_class"] == 3
    assert kwargs["no_prior_continuation_score_clip"] == 6.0


def test_no_prior_continuation_fields_attach_to_full_mht_config() -> None:
    config = full_mht_manifest._full_mht_config_from_options(
        {
            "no_prior_continuation_likelihood_weight": 1.25,
            "no_prior_continuation_min_anchor_registered_iou": 0.72,
            "no_prior_continuation_max_anchor_rank": 1,
            "no_prior_continuation_min_examples_per_class": 3,
            "no_prior_continuation_score_clip": 6.0,
        }
    )

    assert getattr(config, "no_prior_continuation_likelihood_weight") == 1.25
    assert getattr(config, "no_prior_continuation_min_anchor_registered_iou") == 0.72
    assert getattr(config, "no_prior_continuation_max_anchor_rank") == 1
    assert getattr(config, "no_prior_continuation_min_examples_per_class") == 3
    assert getattr(config, "no_prior_continuation_score_clip") == 6.0


def test_no_prior_continuation_probe_manifest_is_frozen() -> None:
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "full_mht_no_prior_continuation_probe_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = {run["name"]: run for run in manifest["runs"]}

    assert list(runs) == [
        "Track2p",
        "FullMHTPrior2",
        "FullMHTCalibratedNoDeath",
        "FullMHTNoPriorContinuation050",
        "FullMHTNoPriorContinuation100",
        "FullMHTNoPriorContinuation150",
    ]
    assert runs["FullMHTCalibratedNoDeath"]["association_score_mode"] == (
        "calibrated-likelihood"
    )
    assert runs["FullMHTCalibratedNoDeath"]["track2p_no_prior_successor_penalty"] == 0.0
    assert runs["FullMHTNoPriorContinuation050"][
        "no_prior_continuation_likelihood_weight"
    ] == 0.5
    assert runs["FullMHTNoPriorContinuation100"][
        "no_prior_continuation_likelihood_weight"
    ] == 1.0
    assert runs["FullMHTNoPriorContinuation150"][
        "no_prior_continuation_likelihood_weight"
    ] == 1.5
    for name in (
        "FullMHTNoPriorContinuation050",
        "FullMHTNoPriorContinuation100",
        "FullMHTNoPriorContinuation150",
    ):
        assert runs[name]["runner"] == "track2p-full-mht"
        assert runs[name]["association_score_mode"] == "calibrated-likelihood"
        assert runs[name]["track2p_no_prior_successor_penalty"] == 0.0
        assert runs[name]["no_prior_continuation_min_examples_per_class"] == 2
