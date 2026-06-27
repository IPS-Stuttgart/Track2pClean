import argparse
import importlib
import importlib.resources
import sys
from types import ModuleType

import pytest
from tests._support import run_module
from track2pclean import _cli as track2pclean_cli


def test_track2pclean_module_help_uses_native_program_name():
    proc = run_module("-m", "track2pclean", "--help")

    assert "usage: track2pclean" in proc.stdout
    assert "track2pclean <command> --help" in proc.stdout
    assert "bayescatrack <command> --help" not in proc.stdout


def test_track2pclean_summary_help_uses_native_program_name():
    proc = run_module("-m", "track2pclean", "summary", "--help")

    assert "usage: track2pclean summary" in proc.stdout


def test_track2pclean_benchmark_help_uses_native_program_name():
    proc = run_module("-m", "track2pclean", "benchmark", "--help")

    assert "usage: track2pclean benchmark" in proc.stdout
    assert "Run Track2pClean benchmark harnesses." in proc.stdout
    assert "bayescatrack benchmark" not in proc.stdout


def test_track2pclean_benchmark_subcommand_help_uses_native_program_name():
    proc = run_module("-m", "track2pclean", "benchmark", "track2p", "--help")

    assert "usage: track2pclean benchmark track2p" in proc.stdout


def test_track2pclean_benchmark_subcommand_inline_help_uses_native_program_name():
    proc = run_module(
        "-m",
        "track2pclean",
        "benchmark",
        "track2p",
        "--data",
        ".",
        "--help",
    )

    assert "usage: track2pclean benchmark track2p" in proc.stdout
    assert "usage: bayescatrack benchmark track2p" not in proc.stdout


def test_track2pclean_benchmark_alias_help_preserves_requested_alias_name():
    proc = run_module(
        "-m",
        "track2pclean",
        "benchmark",
        "audit-manual-gt-roi-index-space",
        "--help",
    )

    assert (
        "usage: track2pclean benchmark audit-manual-gt-roi-index-space" in proc.stdout
    )
    assert "usage: track2pclean benchmark audit-manual-gt-rois" not in proc.stdout


def test_track2pclean_growth_help_uses_native_program_name():
    proc = run_module("-m", "track2pclean", "growth", "--help")

    assert "usage: track2pclean growth" in proc.stdout
    assert "bayescatrack growth" not in proc.stdout


def test_track2pclean_growth_subcommand_help_uses_native_program_name():
    proc = run_module("-m", "track2pclean", "growth", "radial", "--help")

    assert "usage: track2pclean growth radial" in proc.stdout
    assert "bayescatrack growth" not in proc.stdout


def test_track2pclean_advanced_help_uses_native_program_name():
    proc = run_module("-m", "track2pclean", "advanced", "--help")

    assert "usage: track2pclean advanced" in proc.stdout
    assert "Track2pClean result improvement" in proc.stdout
    assert "bayescatrack" not in proc.stdout
    assert "BayesCaTrack" not in proc.stdout


def test_track2pclean_advanced_subcommand_help_uses_native_program_name():
    proc = run_module("-m", "track2pclean", "advanced", "active-labels", "--help")

    assert "usage: track2pclean advanced active-labels" in proc.stdout
    assert "bayescatrack" not in proc.stdout


def test_track2pclean_module_command_forces_native_program_name(monkeypatch, capsys):
    module = ModuleType("tests.fake_track2pclean_module_command")

    def build_arg_parser():
        return argparse.ArgumentParser(prog="unexpected-wrapper")

    def main(args):
        module.build_arg_parser().parse_args(args)
        return 0

    module.build_arg_parser = build_arg_parser
    module.main = main
    monkeypatch.setitem(sys.modules, module.__name__, module)

    with pytest.raises(SystemExit) as exc_info:
        track2pclean_cli._handle_module_command(
            ["--help"],
            module_name=module.__name__,
            program_name="track2pclean fake",
            legacy_program_name="legacy fake",
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 0
    assert "usage: track2pclean fake" in captured.out
    assert "unexpected-wrapper" not in captured.out


def test_track2pclean_is_marked_as_typed_package():
    marker = importlib.resources.files("track2pclean") / "py.typed"

    assert marker.is_file()


def test_bayescatrack_main_module_import_has_no_exit_side_effect():
    module = importlib.import_module("bayescatrack.__main__")

    assert module.main is not None
