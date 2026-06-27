import argparse
import sys
from types import ModuleType

import pytest

from track2pclean import _cli as track2pclean_cli


def test_track2pclean_module_command_retitles_custom_usage(monkeypatch, capsys):
    module = ModuleType("tests.fake_track2pclean_module_custom_usage")

    def build_arg_parser():
        return argparse.ArgumentParser(
            prog="bayescatrack fake",
            usage="bayescatrack fake [--example]",
            description="BayesCaTrack helper from bayescatrack",
        )

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
            legacy_program_name="bayescatrack fake",
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 0
    assert "usage: track2pclean fake [--example]" in captured.out
    assert "bayescatrack fake [--example]" not in captured.out
    assert "BayesCaTrack" not in captured.out
