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


def test_track2pclean_module_command_retitles_argument_group_text(
    monkeypatch, capsys
):
    module = ModuleType("tests.fake_track2pclean_module_argument_group_text")

    def build_arg_parser():
        parser = argparse.ArgumentParser(
            prog="bayescatrack fake",
            description="BayesCaTrack helper from bayescatrack",
        )
        group = parser.add_argument_group(
            title="BayesCaTrack configuration",
            description="Use bayescatrack defaults for grouped options",
        )
        group.add_argument("--example", help="Example grouped option")
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
    assert "Track2pClean configuration" in captured.out
    assert "track2pclean defaults" in captured.out
    assert "BayesCaTrack configuration" not in captured.out
    assert "bayescatrack defaults" not in captured.out


def test_track2pclean_module_command_retitles_subparser_choice_help(
    monkeypatch, capsys
):
    module = ModuleType("tests.fake_track2pclean_module_subparser_choice_help")

    def build_arg_parser():
        parser = argparse.ArgumentParser(
            prog="bayescatrack fake",
            description="BayesCaTrack helper from bayescatrack",
        )
        subparsers = parser.add_subparsers(dest="command", required=True)
        subparsers.add_parser(
            "diagnose",
            help="Run BayesCaTrack diagnostics with bayescatrack defaults",
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
    assert "Run Track2pClean diagnostics with track2pclean defaults" in captured.out
    assert "BayesCaTrack diagnostics" not in captured.out
    assert "bayescatrack defaults" not in captured.out


def test_track2pclean_core_parser_text_retitles_subparser_choice_help(capsys):
    replace_parser_text = getattr(
        track2pclean_cli._replace_parser_text,
        "_track2pclean_original",
        track2pclean_cli._replace_parser_text,
    )
    parser = argparse.ArgumentParser(
        prog="bayescatrack fake",
        description="BayesCaTrack helper from bayescatrack",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "diagnose",
        help="Run BayesCaTrack diagnostics with bayescatrack defaults",
    )

    replace_parser_text(parser, "BayesCaTrack", "Track2pClean")
    replace_parser_text(parser, "bayescatrack", "track2pclean")

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--help"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 0
    assert "Run Track2pClean diagnostics with track2pclean defaults" in captured.out
    assert "BayesCaTrack diagnostics" not in captured.out
    assert "bayescatrack defaults" not in captured.out
