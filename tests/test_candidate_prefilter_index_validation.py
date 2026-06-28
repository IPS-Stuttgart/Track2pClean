from __future__ import annotations

import pytest

from bayescatrack.association.candidate_prefilter import CentroidCandidatePrefilterConfig


class _BadIndex:
    def __index__(self) -> int:
        raise OverflowError("index conversion failed")


@pytest.mark.parametrize("field_name", ["row_top_k", "column_top_k"])
def test_candidate_prefilter_rejects_bad_index_controls(field_name):
    with pytest.raises(ValueError, match=field_name):
        CentroidCandidatePrefilterConfig(**{field_name: _BadIndex()})
