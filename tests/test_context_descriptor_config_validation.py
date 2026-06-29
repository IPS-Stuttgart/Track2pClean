from __future__ import annotations

import numpy as np
import pytest

from bayescatrack.association.context_descriptors import ContextDescriptorConfig


class _BadIndex:
    def __index__(self) -> int:
        raise OverflowError("index conversion failed")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"patch_radius": -1}, "patch_radius"),
        ({"patch_radius": 1.5}, "patch_radius"),
        ({"patch_radius": True}, "patch_radius"),
        ({"patch_radius": _BadIndex()}, "patch_radius"),
        ({"neighbor_k": 0}, "neighbor_k"),
        ({"neighbor_k": 2.5}, "neighbor_k"),
        ({"neighbor_k": False}, "neighbor_k"),
        ({"histogram_bins": 1}, "histogram_bins"),
        ({"histogram_bins": 2.5}, "histogram_bins"),
        ({"density_radius": 0.0}, "density_radius"),
        ({"density_radius": "3.0"}, "density_radius"),
        ({"density_radius": np.inf}, "density_radius"),
    ],
)
def test_context_descriptor_config_rejects_malformed_controls(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ContextDescriptorConfig(**kwargs)


def test_context_descriptor_config_normalizes_numeric_scalar_controls():
    cfg = ContextDescriptorConfig(
        patch_radius=np.int64(2),
        neighbor_k=3.0,
        density_radius=np.float64(4.5),
        histogram_bins=np.int64(6),
    )

    assert cfg.patch_radius == 2
    assert isinstance(cfg.patch_radius, int)
    assert cfg.neighbor_k == 3
    assert isinstance(cfg.neighbor_k, int)
    assert cfg.density_radius == 4.5
    assert cfg.histogram_bins == 6
    assert isinstance(cfg.histogram_bins, int)
