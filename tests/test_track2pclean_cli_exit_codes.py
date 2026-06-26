from tests import _support  # noqa: F401
from track2pclean import _cli as track2pclean_cli


def test_track2pclean_none_delegate_return_maps_to_success():
    def _delegate(args):
        assert args == ["--example"]
        return None

    assert track2pclean_cli._run_with_program_name(
        "track2pclean delegate",
        _delegate,
        ["--example"],
    ) == 0
