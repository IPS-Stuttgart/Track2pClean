from __future__ import annotations

import subprocess
import sys


def test_importing_bayescatrack_does_not_patch_global_argparse() -> None:
    code = """
import argparse
before = argparse.ArgumentParser.add_argument
import bayescatrack  # noqa: F401
after = argparse.ArgumentParser.add_argument
if before is not after:
    raise SystemExit("bayescatrack import patched argparse.ArgumentParser.add_argument")
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
    )
