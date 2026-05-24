from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bayescatrack.experiments import benchmark_manifest as bm
from bayescatrack.experiments import registration_qa_report as rqr
from bayescatrack.experiments.benchmark_manifest import (
    load_benchmark_manifest,
    run_benchmark_manifest,
)


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def test_registration_qa_manifest_uses_runner_specific_writer(tmp_path, monkeypatch):
    """Registration-QA manifest rows need their own Markdown/table writer.

    The generic Track2p benchmark writer only knows the standard benchmark
    metrics columns. A registration-QA manifest run should therefore dispatch
    through the registration-QA output helpers, especially for the table format.
    """

    calls: dict[str, Any] = {}

    def fake_registration_qa_rows(
        config: Any, options: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        calls["config"] = config
        calls["options"] = dict(options)
        return [
            {
                "cost": "registered-iou",
                "registration_backend": "suite2p-affine",
                "transform_type": "affine",
                "edge_count": 1,
            }
        ]

    def fake_backend_writer(
        rows: Sequence[Mapping[str, Any]], output_path: Path, output_format: str
    ) -> None:
        calls["writer"] = "backend-audit"
        calls["writer_rows"] = list(rows)
        calls["writer_format"] = output_format
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("backend audit writer\n", encoding="utf-8")

    monkeypatch.setattr(bm, "_run_registration_qa_rows", fake_registration_qa_rows)
    monkeypatch.setattr(
        rqr, "write_registration_backend_audit_results", fake_backend_writer
    )

    manifest_path = tmp_path / "benchmarks.json"
    _write_manifest(
        manifest_path,
        {
            "runs": [
                {
                    "name": "backend-audit",
                    "runner": "registration-qa",
                    "data": "data",
                    "level": "backend-audit",
                    "format": "table",
                    "output": "results/backend-audit.md",
                }
            ]
        },
    )

    result = run_benchmark_manifest(load_benchmark_manifest(manifest_path))

    assert result.runs[0].rows == 1
    assert calls["options"] == {"level": "backend-audit"}
    assert calls["writer"] == "backend-audit"
    assert calls["writer_format"] == "table"
    assert calls["writer_rows"][0]["registration_backend"] == "suite2p-affine"
    assert (tmp_path / "results" / "backend-audit.md").read_text(
        encoding="utf-8"
    ) == "backend audit writer\n"
