from __future__ import annotations

import numpy as np
import pytest
from bayescatrack.association.postsolve_relinking import PostSolveRelinkingConfig

_NUMERIC_TEXT_BYTES = bytes([49, 46, 48])


@pytest.mark.parametrize(
    "field",
    ["max_edge_cost", "min_cost_improvement", "bidirectional_next_weight"],
)
@pytest.mark.parametrize(
    "bad_value",
    [
        "1.0",
        _NUMERIC_TEXT_BYTES,
        bytearray(_NUMERIC_TEXT_BYTES),
        np.str_("1.0"),
        np.bytes_(_NUMERIC_TEXT_BYTES),
        np.array("1.0", dtype=object),
    ],
)
def test_postsolve_relinking_config_rejects_text_float_controls(
    field: str, bad_value: object
) -> None:
    with pytest.raises(ValueError, match=field):
        PostSolveRelinkingConfig(**{field: bad_value})
