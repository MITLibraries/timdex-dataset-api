# ruff: noqa: SLF001, D205, D209, PLR2004

import datetime
from unittest.mock import patch

import pytest

from timdex_dataset_api import TIMDEXDataset
from timdex_dataset_api.run import TIMDEXRunManager


@pytest.fixture
def timdex_run_manager(dataset_with_runs_location):
    timdex_dataset = TIMDEXDataset(dataset_with_runs_location)
    return TIMDEXRunManager(timdex_dataset=timdex_dataset)


def test_timdex_run_manager_init(dataset_with_runs_location):
    timdex_dataset = TIMDEXDataset(dataset_with_runs_location)
    timdex_run_manager = TIMDEXRunManager(timdex_dataset=timdex_dataset)
    assert timdex_run_manager._runs_metadata_cache is None


def test_timdex_run_manager_parse_single_parquet_file_success(timdex_run_manager):
    """Parse run metadata from first parquet file in fixture dataset.  We know the details
    of this ETL run in advance given the deterministic fixture that generated it."""
    parquet_filepath = timdex_run_manager.timdex_dataset.dataset.files[0]
    run_metadata = timdex_run_manager._parse_run_metadata_from_parquet_file(
        parquet_filepath
    )
    assert run_metadata["source"] == "alma"
    assert run_metadata["run_date"] == datetime.date(2024, 12, 1)
    assert run_metadata["run_type"] == "full"
    assert run_metadata["run_id"] == "run-1"
    assert run_metadata["num_rows"] == 40
    assert run_metadata["filename"] == parquet_filepath


def test_timdex_run_manager_parse_multiple_parquet_files(timdex_run_manager):
    parquet_metadata_df = timdex_run_manager._get_parquet_files_run_metadata()

    # assert 16 rows for this per-file dataframe, despite only 14 distinct ETL "runs"
    assert len(parquet_metadata_df) == 16

    # assert each source has metadata for 8 parquet files
    assert parquet_metadata_df.source.value_counts().to_dict() == {"alma": 8, "dspace": 8}


def test_timdex_run_manager_get_runs_df(timdex_run_manager):
    runs_df = timdex_run_manager.get_runs_metadata()

    # assert two "large" runs have multiple parquet files
    assert len(runs_df[runs_df.parquet_files_count > 1]) == 2

    # assert 7 distinct runs per source, despite more parquet files
    assert runs_df.source.value_counts().to_dict() == {"alma": 7, "dspace": 7}


def test_timdex_run_manager_get_all_current_run_parquet_files_success(
    timdex_run_manager,
):
    ordered_parquet_files = timdex_run_manager.get_current_parquet_files()

    # assert 12 parquet files, despite being 14 total for ALL sources
    # this represents the last full run and all daily since
    assert len(ordered_parquet_files) == 12

    # assert sorted reverse chronologically
    assert "year=2025/month=01/day=01" in ordered_parquet_files[-1]


def test_timdex_run_manager_get_source_current_run_parquet_files_success(
    timdex_run_manager,
):
    ordered_parquet_files = timdex_run_manager._get_current_source_parquet_files("alma")

    # assert 6 parquet files, despite being 8 total for 'alma' source
    # this represents the last full run and all daily since
    assert len(ordered_parquet_files) == 6

    # assert sorted reverse chronologically
    assert "year=2025/month=01/day=05" in ordered_parquet_files[0]
    assert "year=2025/month=01/day=01" in ordered_parquet_files[-1]


def test_timdex_run_manager_caches_runs_dataframe(timdex_run_manager):
    runs_df = timdex_run_manager.get_runs_metadata()
    assert timdex_run_manager._runs_metadata_cache is not None

    with patch.object(
        timdex_run_manager, "_get_parquet_files_run_metadata"
    ) as mocked_intermediate_method:
        mocked_intermediate_method.side_effect = Exception(
            "I am not reached, cache is used."
        )
        runs_df_2 = timdex_run_manager.get_runs_metadata()

    assert runs_df.equals(runs_df_2)
