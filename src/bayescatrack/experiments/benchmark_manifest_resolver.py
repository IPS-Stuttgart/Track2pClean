"""Resolve benchmark suite manifest template placeholders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from bayescatrack.experiments.benchmark_manifest import load_benchmark_manifest

_PLACEHOLDERS = {
    "<DATA_ROOT>": "data_root",
    "<REFERENCE_ROOT>": "reference_root",
    "<OUTPUT_ROOT>": "output_root",
}


def resolve_benchmark_manifest_placeholders(
    manifest_path: str | Path,
    *,
    data_root: str | Path,
    reference_root: str | Path,
    output_root: str | Path,
    output: str | Path,
) -> Path:
    """Write a manifest copy with standard root placeholders replaced."""

    path = Path(manifest_path)
    raw_manifest = _load_json(path)
    replacements = {
        "<DATA_ROOT>": str(data_root),
        "<REFERENCE_ROOT>": str(reference_root),
        "<OUTPUT_ROOT>": str(output_root),
    }
    resolved_manifest = _replace_placeholders(raw_manifest, replacements)

    unresolved = sorted(_find_known_placeholders(resolved_manifest))
    if unresolved:
        raise ValueError(
            "Resolved manifest still contains placeholders: " + ", ".join(unresolved)
        )

    output_path = Path(output)
    if output_path.resolve() == path.resolve():
        raise ValueError("Resolved manifest output must differ from the template path")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(resolved_manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    # Parse through the suite loader so malformed manifests fail before a run starts.
    load_benchmark_manifest(output_path)
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the manifest resolver CLI parser."""

    parser = argparse.ArgumentParser(
        prog="bayescatrack benchmark resolve-suite",
        description="Resolve benchmark suite manifest root placeholders.",
    )
    parser.add_argument("manifest", type=Path, help="Template JSON benchmark manifest")
    parser.add_argument(
        "--data-root",
        required=True,
        help="Replacement for <DATA_ROOT>",
    )
    parser.add_argument(
        "--reference-root",
        required=True,
        help="Replacement for <REFERENCE_ROOT>",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Replacement for <OUTPUT_ROOT>",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the resolved JSON benchmark manifest",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the manifest resolver CLI."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    output_path = resolve_benchmark_manifest_placeholders(
        args.manifest,
        data_root=args.data_root,
        reference_root=args.reference_root,
        output_root=args.output_root,
        output=args.output,
    )
    print(json.dumps({"output": str(output_path)}, indent=2))
    return 0


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _replace_placeholders(value: Any, replacements: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        resolved = value
        for placeholder, replacement in replacements.items():
            resolved = resolved.replace(placeholder, replacement)
        return resolved
    if isinstance(value, list):
        return [_replace_placeholders(item, replacements) for item in value]
    if isinstance(value, dict):
        return {
            _replace_placeholders(key, replacements): _replace_placeholders(
                item, replacements
            )
            for key, item in value.items()
        }
    return value


def _find_known_placeholders(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, str):
        found.update(placeholder for placeholder in _PLACEHOLDERS if placeholder in value)
    elif isinstance(value, list):
        for item in value:
            found.update(_find_known_placeholders(item))
    elif isinstance(value, dict):
        for key, item in value.items():
            found.update(_find_known_placeholders(key))
            found.update(_find_known_placeholders(item))
    return found


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
