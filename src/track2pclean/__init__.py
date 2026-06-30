"""Track2pClean public compatibility namespace.

The implementation still lives in :mod:`bayescatrack` to avoid breaking existing
result manifests, notebooks, and downstream imports. New user-facing
installations expose the ``track2pclean`` console script, while historical
``bayescatrack`` imports continue to work.
"""

from bayescatrack import *  # noqa: F401,F403
from bayescatrack import __all__ as _bayescatrack_all
from bayescatrack._cli_exit_code_validation import (
    _coerce_exit_code as _strict_exit_code_coerce,
)
from bayescatrack._fov_translation_bytes_shape_validation import (
    install_fov_translation_bytes_shape_validation as _install_fov_translation_bytes_shape_validation,
)

from . import _cli as _cli
from ._custom_usage_text_validation import (
    install_custom_usage_text_validation as _install_custom_usage_text_validation,
)

_cli._coerce_exit_code = _strict_exit_code_coerce  # pylint: disable=protected-access
_install_fov_translation_bytes_shape_validation()
_install_custom_usage_text_validation(_cli)
main = _cli.main

__all__ = tuple(dict.fromkeys((*_bayescatrack_all, "main")))
