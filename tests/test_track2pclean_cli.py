import importlib
import importlib.resources

from tests._support import run_module


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


def test_track2pclean_is_marked_as_typed_package():
    marker = importlib.resources.files("track2pclean") / "py.typed"

    assert marker.is_file()


def test_bayescatrack_main_module_import_has_no_exit_side_effect():
    module = importlib.import_module("bayescatrack.__main__")

    assert module.main is not None
