import argparse
import importlib.util
import sys
from types import ModuleType

import pytest

from track2pclean import _cli as track2pclean_cli


def _load_unpatched_track2pclean_cli_module():
    module_name = "tests.unpatched_track2pclean_cli"
    module_path = track2pclean_cli.__file__
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.pop(module_name, None)
    spec.loader.exec_module(module)
    return module


def test_track2pclean_native_retitle_rewrites_custom_usage_without_import_hook():
    native_cli = _load_unpatched_track2pclean_cli_module()
    parser = argparse.ArgumentParser(
        prog="bayescatrack fake",
        usage="bayescatrack fake [--example]",
        description="BayesCaTrack helper from bayescatrack",
    )

    native_cli._retitle_arg_parser(
        parser,
        legacy_program_name="bayescatrack fake",
        program_name="track2pclean fake",
    )

    help_text = parser.format_help()
    assert "usage: track2pclean fake [--example]" in help_text
    assert "bayescatrack fake [--example]" not in help_text
    assert "BayesCaTrack" not in help_text


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
