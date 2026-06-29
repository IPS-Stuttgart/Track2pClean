from __future__ import annotations

import pytest
from bayescatrack.core import bridge as bridge_module


def test_track2pclean_summary_parser_accepts_strict(tmp_path):
    parser = bridge_module._bridge_impl._build_arg_parser()  # pylint: disable=protected-access
    args = parser.parse_args(["summary", str(tmp_path), "--strict"])
    assert args.strict is True
