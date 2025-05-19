# ruff: noqa: SLF001, D205, D209, PLR2004

import pytest

from timdex_dataset_api import TIMDEXDataset, TIMDEXSource


@pytest.fixture
def timdex_source_alma(dataset_with_runs_location) -> TIMDEXSource:
    timdex_source = TIMDEXSource(dataset_with_runs_location, "alma")
    timdex_source.load()
    return timdex_source


def test_timdex_source_init(dataset_with_runs_location):
    timdex_source = TIMDEXSource(dataset_with_runs_location, "alma")

    # assert TIMDEXSource is a subclass of TIMDEXDataset
    assert isinstance(timdex_source, TIMDEXDataset)

    # assert a  TIMDEXSource instance requires an explicit source set
    assert timdex_source.source == "alma"

    # assert "seen" records is an empty set on init
    assert len(timdex_source._seen_records) == 0


def test_timdex_source_load_success(dataset_with_runs_location):
    timdex_source = TIMDEXSource(dataset_with_runs_location, "alma")
    timdex_source.load()

    # assert dataset is loaded with parquet files associated with current runs only
    assert len(timdex_source.dataset.files) == 6
    assert len(timdex_source.paths) == 6


def test_timdex_source_yields_reverse_chronologically_by_default(
    timdex_source_alma,
):
    """Tests that each yielded record is less than or equal to the previous record."""
    last_date = None
    for record in timdex_source_alma.read_dicts_iter():
        assert last_date is None or record["run_date"] <= last_date
        last_date = record["run_date"]


def test_timdex_source_yields_only_current_records(
    timdex_source_alma,
):
    full_df = timdex_source_alma.read_dataframe()

    # assert records are deduplicated
    assert full_df.timdex_record_id.is_unique

    # assert 100 records in dataframe, despite 100+ in dataset for source='alma'
    assert len(full_df) == 100

    # assert that 99 records are "to-index" and one is "to-delete" based on order of runs
    assert full_df.action.value_counts().to_dict() == {"index": 99, "delete": 1}


def test_timdex_source_all_read_methods_get_deduplication_and_ordering(
    timdex_source_alma,
):
    full_df = timdex_source_alma.read_dataframe()
    all_records = list(timdex_source_alma.read_dicts_iter())
    transformed_records = list(timdex_source_alma.read_transformed_records_iter())
    assert len(full_df) == len(all_records) == len(transformed_records)
