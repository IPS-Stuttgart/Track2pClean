import sys
from types import ModuleType

import numpy as np
import pytest

from tests import _support  # noqa: F401
from bayescatrack import cli as bayescatrack_cli
from track2pclean import _cli as track2pclean_cli


_INVALID_EXIT_CODE_RETURNS = [
    True,
    False,
    np.bool_(True),
    np.array(0),
    np.array(7, dtype=np.int64),
    "0",
    b"1",
    1.0,
    1.5,
    object(),
]

_OUT_OF_RANGE_EXIT_CODE_RETURNS = [-1, 256, np.int64(256)]


def test_track2pclean_none_delegate_return_maps_to_success():
    def _delegate(args):
        assert args == ["--example"]
        return None

    assert track2pclean_cli._run_with_program_name(
        "track2pclean delegate",
        _delegate,
        ["--example"],
    ) == 0


@pytest.mark.parametrize(
    ("delegate_result", "expected_exit_code"),
    [(0, 0), (2, 2), (np.int64(3), 3)],
)
def test_track2pclean_integer_delegate_return_maps_to_exit_code(
    delegate_result,
    expected_exit_code,
):
    def _delegate(args):
        assert args == ["--example"]
        return delegate_result

    assert track2pclean_cli._run_with_program_name(
        "track2pclean delegate",
        _delegate,
        ["--example"],
    ) == expected_exit_code


@pytest.mark.parametrize("delegate_result", _INVALID_EXIT_CODE_RETURNS)
def test_track2pclean_rejects_non_integer_delegate_return(delegate_result):
    def _delegate(args):
        assert args == ["--example"]
        return delegate_result

    with pytest.raises(TypeError, match="integer exit code"):
        track2pclean_cli._run_with_program_name(
            "track2pclean delegate",
            _delegate,
            ["--example"],
        )


@pytest.mark.parametrize("delegate_result", _OUT_OF_RANGE_EXIT_CODE_RETURNS)
def test_track2pclean_rejects_out_of_range_integer_delegate_return(delegate_result):
    def _delegate(args):
        assert args == ["--example"]
        return delegate_result

    with pytest.raises(ValueError, match="integer exit code"):
        track2pclean_cli._run_with_program_name(
            "track2pclean delegate",
            _delegate,
            ["--example"],
        )


def test_bayescatrack_core_delegate_none_return_maps_to_success(monkeypatch):
    observed_args = []

    def _delegate(args):
        observed_args.append(args)
        return None

    monkeypatch.setattr(bayescatrack_cli, "_core_main", _delegate)

    assert bayescatrack_cli.main(["summary", "--example"]) == 0
    assert observed_args == [["summary", "--example"]]


@pytest.mark.parametrize("delegate_result", _INVALID_EXIT_CODE_RETURNS)
def test_bayescatrack_core_rejects_non_integer_delegate_return(
    delegate_result,
    monkeypatch,
):
    def _delegate(args):
        assert args == ["summary", "--example"]
        return delegate_result

    monkeypatch.setattr(bayescatrack_cli, "_core_main", _delegate)

    with pytest.raises(TypeError, match="integer exit code"):
        bayescatrack_cli.main(["summary", "--example"])


@pytest.mark.parametrize("delegate_result", _OUT_OF_RANGE_EXIT_CODE_RETURNS)
def test_bayescatrack_core_rejects_out_of_range_integer_delegate_return(
    delegate_result,
    monkeypatch,
):
    def _delegate(args):
        assert args == ["summary", "--example"]
        return delegate_result

    monkeypatch.setattr(bayescatrack_cli, "_core_main", _delegate)

    with pytest.raises(ValueError, match="integer exit code"):
        bayescatrack_cli.main(["summary", "--example"])


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
