import sys
from types import ModuleType

from tests import _support  # noqa: F401
from bayescatrack import cli as bayescatrack_cli
from track2pclean import _cli as track2pclean_cli


def test_track2pclean_none_delegate_return_maps_to_success():
    def _delegate(args):
        assert args == ["--example"]
        return None

    assert track2pclean_cli._run_with_program_name(
        "track2pclean delegate",
        _delegate,
        ["--example"],
    ) == 0


def test_bayescatrack_core_delegate_none_return_maps_to_success(monkeypatch):
    observed_args = []

    def _delegate(args):
        observed_args.append(args)
        return None

    monkeypatch.setattr(bayescatrack_cli, "_core_main", _delegate)

    assert bayescatrack_cli.main(["summary", "--example"]) == 0
    assert observed_args == [["summary", "--example"]]


def test_bayescatrack_benchmark_delegate_none_return_maps_to_success(monkeypatch):
    observed_args = []
    fake_module = ModuleType("tests.none_returning_benchmark")

    def _delegate(args):
        observed_args.append(args)
        return None

    fake_module.main = _delegate
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)
    monkeypatch.setitem(
        bayescatrack_cli._BENCHMARK_COMMANDS,
        "none-returning",
        bayescatrack_cli._BenchmarkCommand(
            fake_module.__name__,
            "No-op benchmark",
        ),
    )

    assert bayescatrack_cli.main(["benchmark", "none-returning", "--example"]) == 0
    assert observed_args == [["--example"]]
