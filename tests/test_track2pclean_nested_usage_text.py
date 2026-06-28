import argparse

from track2pclean import _cli as track2pclean_cli


def test_track2pclean_retitles_nested_custom_usage_text():
    parser = argparse.ArgumentParser(
        prog="bayescatrack parent",
        description="BayesCaTrack helper from bayescatrack",
    )
    subparsers = parser.add_subparsers(dest="command")
    child_parser = subparsers.add_parser(
        "child",
        prog="bayescatrack parent child",
        usage="bayescatrack parent child [--example]",
        description="BayesCaTrack child helper from bayescatrack",
    )

    track2pclean_cli._retitle_arg_parser(
        parser,
        legacy_program_name="bayescatrack parent",
        program_name="track2pclean parent",
    )

    help_text = child_parser.format_help()
    assert "usage: track2pclean parent child [--example]" in help_text
    assert "bayescatrack parent child [--example]" not in help_text
    assert "BayesCaTrack" not in help_text
