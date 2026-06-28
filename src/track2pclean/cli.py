"""Public Track2pClean command-line module.

This module mirrors the historical ``bayescatrack.cli`` public import path and
keeps the native Track2pClean implementation in :mod:`track2pclean._cli`.
"""

from __future__ import annotations

from track2pclean._cli import main

__all__ = ["main"]

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
