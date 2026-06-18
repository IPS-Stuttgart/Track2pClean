from __future__ import annotations

import re
from pathlib import Path

from bayescatrack.dependency_pins import (
    PYRECEST_COMMIT,
    PYRECEST_DIRECT_URL,
    PYRECEST_REPOSITORY,
)


def test_pyrecest_dependency_pin_matches_project_metadata():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    match = re.search(
        r"pyrecest\s*@\s*git\+https://github\.com/FlorianPfaff/PyRecEst\.git@(?P<commit>[0-9a-f]{40})",
        pyproject,
    )

    assert match is not None
    assert match.group("commit") == PYRECEST_COMMIT
    assert PYRECEST_REPOSITORY == "https://github.com/FlorianPfaff/PyRecEst.git"
    assert PYRECEST_DIRECT_URL.endswith(f"@{PYRECEST_COMMIT}")


def test_scikit_image_is_declared_for_threshold_benchmark_imports():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert "scikit-image" in pyproject
    assert "scikit-image>=0.25.2; python_version >= '3.13'" in pyproject
