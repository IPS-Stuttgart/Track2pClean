"""Track2pClean public compatibility namespace.

The implementation still lives in :mod:`bayescatrack` to avoid breaking existing
result manifests, notebooks, and downstream imports. New user-facing
installations expose the ``track2pclean`` console script, while historical
``bayescatrack`` imports continue to work.
"""

from bayescatrack import *  # noqa: F401,F403
from bayescatrack import __all__ as __all__, main as main
