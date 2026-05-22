import json
from pathlib import Path

from bayescatrack.cli import _handle_benchmark
from bayescatrack.experiments.benchmark_manifest import load_benchmark_manifest
from bayescatrack.experiments.benchmark_manifest_plan import build_manifest_plan


def test_track2p_policy_cli_dispatch(monkeypatch) -> None:
    from bayescatrack.experiments import track2p_policy_benchmark

    calls: list[tuple[str, ...]] = []

    def fake_main(argv: list[str] | None = None) -> int:
        calls.append(tuple(argv or ()))
        return 17

    monkeypatch.setattr(track2p_policy_benchmark, "main", fake_main)

    assert _handle_benchmark(["track2p-policy", "--sentinel"]) == 17
    assert calls == [("--sentinel",)]


def test_track2p_policy_dp_cli_dispatch(monkeypatch) -> None:
    from bayescatrack.experiments import track2p_policy_dp_benchmark

    calls: list[tuple[str, ...]] = []

    def fake_main(argv: list[str] | None = None) -> int:
        calls.append(tuple(argv or ()))
        return 19

    monkeypatch.setattr(track2p_policy_dp_benchmark, "main", fake_main)

    assert _handle_benchmark(["track2p-policy-dp", "--sentinel"]) == 19
    assert calls == [("--sentinel",)]


def test_manifest_accepts_track2p_policy_runners(tmp_path: Path) -> None:
    manifest_path = tmp_path / "suite.json"
    manifest_path.write_text(
        json.dumps(
            {
                "defaults": {
                    "data": "data",
                    "reference_kind": "manual-gt",
                    "input_format": "suite2p",
                },
                "runs": [
                    {
                        "name": "policy",
                        "runner": "track2p-policy",
                        "threshold_method": "min",
                        "iou_distance_threshold": 12.0,
                    },
                    {
                        "name": "policy-dp",
                        "runner": "track2p-policy-dp",
                        "threshold_method": "otsu",
                        "iou_distance_threshold": 16.0,
                        "row_top_k": 3,
                        "beam_width": 5,
                        "max_gap": 2,
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = load_benchmark_manifest(manifest_path)

    assert manifest.runs[0].runner == "track2p-policy"
    assert manifest.runs[0].config.method == "global-assignment"
    assert manifest.runs[0].runner_kwargs == {
        "threshold_method": "min",
        "iou_distance_threshold": 12.0,
    }
    assert manifest.runs[1].runner == "track2p-policy-dp"
    assert manifest.runs[1].config.method == "global-assignment"
    assert manifest.runs[1].config.max_gap == 2
    assert manifest.runs[1].runner_kwargs["row_top_k"] == 3

    plan = build_manifest_plan(manifest)
    assert plan["runs"][0]["runner_option_keys"] == [
        "iou_distance_threshold",
        "threshold_method",
    ]
    assert "row_top_k" in plan["runs"][1]["runner_option_keys"]


def test_guarded_benchmark_script_includes_policy_runs() -> None:
    script = Path(".github/scripts/run_track2p_benchmark.py").read_text(encoding="utf-8")

    assert '\"runner\": \"track2p-policy\"' in script
    assert '\"runner\": \"track2p-policy-dp\"' in script
    assert '\"Track2p policy\": \"track2p-policy\"' in script
    assert '\"Track2p policy DP\": \"track2p-policy-dp\"' in script
