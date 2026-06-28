from tests import _support  # noqa: F401


def test_raw_benchmark_bare_string_exclude_subject_is_single_subject(tmp_path):
    from bayescatrack.experiments.track2p_raw_benchmark_data import (
        prepare_raw_suite2p_benchmark_data,
    )

    source_root = tmp_path / "source"
    (source_root / "jm001" / "track2p").mkdir(parents=True)

    summary = prepare_raw_suite2p_benchmark_data(
        raw_root=source_root,
        metadata_root=source_root,
        output_root=tmp_path / "prepared",
        exclude_subjects="jm001",
        min_subjects=0,
    )

    assert summary.excluded_by_user == ("jm001",)
    assert summary.excluded_no_raw_suite2p == ()
