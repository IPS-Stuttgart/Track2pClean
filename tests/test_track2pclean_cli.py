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
