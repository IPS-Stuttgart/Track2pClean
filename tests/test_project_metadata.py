from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_project_metadata_matches_repository():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert project["name"] == "Track2pClean"
    assert project["urls"]["Repository"] == "https://github.com/IPS-Stuttgart/Track2pClean"
    assert project["scripts"]["bayescatrack"] == "bayescatrack:main"
    assert project["scripts"]["track2pclean"] == "track2pclean:main"
