"""Retitle custom argparse usage text in the Track2pClean CLI wrapper.

The native wrapper already retitles parser descriptions, epilogues, action help,
and derived ``prog`` values.  Delegated parsers that set a custom ``usage``
string need the same rewrite so ``track2pclean ... --help`` does not leak the
legacy ``bayescatrack`` command name.
"""

from __future__ import annotations

from typing import Any

_PATCH_MARKER = "_track2pclean_custom_usage_text_validation_patch"


def install_custom_usage_text_validation(cli_module: Any) -> None:
    """Install idempotent retitling for ``ArgumentParser.usage`` strings."""

    original_replace_parser_text = (
        cli_module._replace_parser_text
    )  # pylint: disable=protected-access
    if getattr(original_replace_parser_text, _PATCH_MARKER, False):
        return

    def _replace_parser_text_with_usage(
        parser: Any,
        old_text: str,
        new_text: str,
    ) -> None:
        original_replace_parser_text(parser, old_text, new_text)
        usage = getattr(parser, "usage", None)
        if isinstance(usage, str):
            parser.usage = usage.replace(old_text, new_text)

    setattr(_replace_parser_text_with_usage, _PATCH_MARKER, True)
    setattr(
        _replace_parser_text_with_usage,
        "_track2pclean_original",
        original_replace_parser_text,
    )
    cli_module._replace_parser_text = (
        _replace_parser_text_with_usage  # pylint: disable=protected-access
    )


__all__ = ["install_custom_usage_text_validation"]
