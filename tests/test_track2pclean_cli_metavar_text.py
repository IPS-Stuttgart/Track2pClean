import argparse
import sys
from types import ModuleType

import pytest
from track2pclean import _cli as track2pclean_cli


def test_track2pclean_module_command_retitles_action_metavar(monkeypatch, capsys):
    module = ModuleType("tests.fake_track2pclean_module_action_metavar")

    def build_arg_parser():
        parser = argparse.ArgumentParser(
            prog="bayescatrack fake",
            description="BayesCaTrack helper from bayescatrack",
        )
        parser.add_argument(
            "--config",
            metavar="bayescatrack-config",
            help="Path to a BayesCaTrack config",
        )
        return parser

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
    assert "track2pclean-config" in captured.out
    assert "bayescatrack-config" not in captured.out
    assert "Path to a Track2pClean config" in captured.out
