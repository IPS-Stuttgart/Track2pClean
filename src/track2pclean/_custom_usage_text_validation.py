"""Retitle auxiliary argparse text in the Track2pClean CLI wrapper.

The native wrapper retitles parser descriptions, epilogues, action help, usage,
and derived ``prog`` values. Delegated parsers can still hide user-visible text
in argparse's subparser choice pseudo-actions. This patch keeps those choice
help rows and explicit usage strings under the native Track2pClean name so
``track2pclean ... --help`` does not leak the legacy ``bayescatrack`` command
name.
"""

from __future__ import annotations

import argparse
from typing import Any

_PATCH_MARKER = "_track2pclean_custom_usage_text_validation_patch"


def install_custom_usage_text_validation(cli_module: Any) -> None:
    """Install idempotent retitling for auxiliary argparse help text."""

    original_replace_parser_text = (
        cli_module._replace_parser_text
    )  # pylint: disable=protected-access
    if getattr(original_replace_parser_text, _PATCH_MARKER, False):
        return

    def _replace_parser_text_with_auxiliary_help(
        parser: argparse.ArgumentParser,
        old_text: str,
        new_text: str,
    ) -> None:
        original_replace_parser_text(parser, old_text, new_text)
        _replace_parser_usage_text(
            parser,
            old_text=old_text,
            new_text=new_text,
            cli_module=cli_module,
        )
        _replace_subparser_choice_help_text(
            parser,
            old_text=old_text,
            new_text=new_text,
            cli_module=cli_module,
        )

    setattr(_replace_parser_text_with_auxiliary_help, _PATCH_MARKER, True)
    setattr(
        _replace_parser_text_with_auxiliary_help,
        "_track2pclean_original",
        original_replace_parser_text,
    )
    cli_module._replace_parser_text = (  # pylint: disable=protected-access
        _replace_parser_text_with_auxiliary_help
    )


def _replace_parser_usage_text(
    parser: argparse.ArgumentParser,
    *,
    old_text: str,
    new_text: str,
    cli_module: Any,
) -> None:
    """Recursively rewrite explicit ``usage`` strings on parser trees."""

    usage = getattr(parser, "usage", None)
    if isinstance(usage, str):
        parser.usage = usage.replace(old_text, new_text)

    for child_parser in cli_module._iter_child_arg_parsers(  # pylint: disable=protected-access
        parser
    ):
        _replace_parser_usage_text(
            child_parser,
            old_text=old_text,
            new_text=new_text,
            cli_module=cli_module,
        )


def _replace_subparser_choice_help_text(
    parser: argparse.ArgumentParser,
    *,
    old_text: str,
    new_text: str,
    cli_module: Any,
) -> None:
    """Recursively rewrite help stored in argparse subparser choice rows."""

    for action in parser._actions:  # pylint: disable=protected-access
        for choice_action in getattr(action, "_choices_actions", ()):  # noqa: SLF001
            help_text = getattr(choice_action, "help", None)
            if isinstance(help_text, str):
                choice_action.help = help_text.replace(old_text, new_text)

    for child_parser in cli_module._iter_child_arg_parsers(  # pylint: disable=protected-access
        parser
    ):
        _replace_subparser_choice_help_text(
            child_parser,
            old_text=old_text,
            new_text=new_text,
            cli_module=cli_module,
        )


__all__ = ["install_custom_usage_text_validation"]
