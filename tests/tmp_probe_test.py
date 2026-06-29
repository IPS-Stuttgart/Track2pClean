import pytest


def test_probe():
    with pytest.raises(ValueError):
        int('x')
