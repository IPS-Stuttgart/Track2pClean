"""Regression tests for package import side effects."""

from __future__ import annotations

import argparse
import importlib
from typing import Any


def test_package_import_does_not_replace_argparse_add_argument(monkeypatch) -> None:
    """Importing BayesCaTrack must not monkey-patch argparse globally."""

    original_add_argument = argparse.ArgumentParser.add_argument

    def sentinel_add_argument(
        self: argparse.ArgumentParser, *args: Any, **kwargs: Any
    ) -> Any:
        return original_add_argument(self, *args, **kwargs)

    monkeypatch.setattr(argparse.ArgumentParser, "add_argument", sentinel_add_argument)

    module = importlib.import_module("bayescatrack")
    importlib.reload(module)

    assert argparse.ArgumentParser.add_argument is sentinel_add_argument
