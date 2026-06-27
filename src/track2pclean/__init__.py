"""Track2pClean public compatibility namespace.

The implementation still lives in :mod:`bayescatrack` to avoid breaking existing
result manifests, notebooks, and downstream imports. New user-facing
installations expose the ``track2pclean`` console script, while historical
``bayescatrack`` imports continue to work.
"""

from bayescatrack import *  # noqa: F401,F403
from bayescatrack import __all__ as _bayescatrack_all

from . import _cli as _cli
from ._custom_usage_text_validation import (
    install_custom_usage_text_validation as _install_custom_usage_text_validation,
)

_install_custom_usage_text_validation(_cli)
main = _cli.main

__all__ = tuple(dict.fromkeys((*_bayescatrack_all, "main")))
