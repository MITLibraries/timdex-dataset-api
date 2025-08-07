# ruff: noqa: D205, D209, SLF001, PLR2004

import glob
import os
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pytest
from pyarrow import fs

from timdex_dataset_api.dataset import (
    TIMDEXDataset,
    TIMDEXDatasetConfig,
)


def test_dataset_init_success(tmp_path):
    timdex_dataset = TIMDEXDataset(str(tmp_path / "path/to/dataset"))
    assert isinstance(timdex_dataset.dataset.filesystem, fs.LocalFileSystem)


def test_dataset_init_env_vars_set_config(monkeypatch, tmp_path):
    location = str(tmp_path / "timdex_dataset/")
    default_timdex_dataset = TIMDEXDataset(location=location)
    default_read_batch_config = default_timdex_dataset.config.read_batch_size
    assert default_read_batch_config == 1_000

    monkeypatch.setenv("TDA_READ_BATCH_SIZE", "100_000")
    env_var_timdex_dataset = TIMDEXDataset(location=location)
    env_var_read_batch_config = env_var_timdex_dataset.config.read_batch_size
    assert env_var_read_batch_config == 100_000


def test_dataset_init_custom_config_object(monkeypatch, tmp_path):
    location = str(tmp_path / "timdex_dataset/")
    config = TIMDEXDatasetConfig()
    config.max_rows_per_file = 42
    timdex_dataset = TIMDEXDataset(location=location, config=config)
    assert timdex_dataset.config.max_rows_per_file == 42


@patch("timdex_dataset_api.dataset.fs.LocalFileSystem")
@patch("timdex_dataset_api.dataset.ds.dataset")
def test_load_pyarrow_dataset_default_uses_data_records_root(
    mock_pyarrow_ds, mock_local_fs, tmp_path
):
    """Ensure load_pyarrow_dataset() without args calls pyarrow.dataset with the
    dataset's data_records_root path as the source and the proper filesystem."""
    mock_local_fs.return_value = MagicMock()
    mock_pyarrow_ds.return_value = MagicMock()

    location = str(Path(tmp_path) / "local/path/to/default_dataset")

    timdex_dataset = TIMDEXDataset(location=location)
    # call the explicit loader to exercise the code path
    dataset_obj = timdex_dataset.load_pyarrow_dataset()

    mock_pyarrow_ds.assert_called_with(
        f"{location}/data/records",
        schema=timdex_dataset.schema,
        format="parquet",
        partitioning="hive",
        filesystem=mock_local_fs.return_value,
    )
    assert dataset_obj == mock_pyarrow_ds.return_value
    assert timdex_dataset.dataset == mock_pyarrow_ds.return_value


@patch("timdex_dataset_api.dataset.fs.LocalFileSystem")
@patch("timdex_dataset_api.dataset.ds.dataset")
def test_load_pyarrow_dataset_with_parquet_files_list(
    mock_pyarrow_ds, mock_local_fs, tmp_path
):
    """Ensure load_pyarrow_dataset(parquet_files=...) passes the explicit list
    of parquet files as the source to pyarrow.dataset."""
    mock_local_fs.return_value = MagicMock()
    mock_pyarrow_ds.return_value = MagicMock()

    location = str(Path(tmp_path) / "local/path/to/dataset_with_files")

    timdex_dataset = TIMDEXDataset(location=location)

    parquet_files = [
        f"{timdex_dataset.data_records_root}/source=alma/run_date=2024-12-01/part-0.parquet",
        f"{timdex_dataset.data_records_root}/source=alma/run_date=2024-12-01/part-1.parquet",
    ]

    dataset_obj = timdex_dataset.load_pyarrow_dataset(parquet_files=parquet_files)

    mock_pyarrow_ds.assert_called_with(
        parquet_files,
        schema=timdex_dataset.schema,
        format="parquet",
        partitioning="hive",
        filesystem=mock_local_fs.return_value,
    )
    assert dataset_obj == mock_pyarrow_ds.return_value
    assert timdex_dataset.dataset == mock_pyarrow_ds.return_value


@patch("timdex_dataset_api.dataset.fs.LocalFileSystem")
@patch("timdex_dataset_api.dataset.ds.dataset")
def test_dataset_load_local_sets_filesystem_and_dataset_success(
    mock_pyarrow_ds, mock_local_fs, tmp_path
):
    mock_local_fs.return_value = MagicMock()
    mock_pyarrow_ds.return_value = MagicMock()

    location = str(Path(tmp_path) / "local/path/to/dataset")

    timdex_dataset = TIMDEXDataset(location=location)

    mock_pyarrow_ds.assert_called_once_with(
        f"{location}/data/records",
        schema=timdex_dataset.schema,
        format="parquet",
        partitioning="hive",
        filesystem=mock_local_fs.return_value,
    )

    assert timdex_dataset.dataset == mock_pyarrow_ds.return_value


@patch("timdex_dataset_api.dataset.TIMDEXDataset.get_s3_filesystem")
@patch("timdex_dataset_api.dataset.ds.dataset")
def test_dataset_load_s3_sets_filesystem_and_dataset_success(
    mock_pyarrow_ds, mock_get_s3_fs, s3_bucket_mocked
):
    mock_get_s3_fs.return_value = MagicMock()
    mock_pyarrow_ds.return_value = MagicMock()

    timdex_dataset = TIMDEXDataset(location="s3://timdex/path/to/dataset")

    mock_pyarrow_ds.assert_called_with(
        "timdex/path/to/dataset/data/records",
        schema=timdex_dataset.schema,
        format="parquet",
        partitioning="hive",
        filesystem=mock_get_s3_fs.return_value,
    )
    assert timdex_dataset.dataset == mock_pyarrow_ds.return_value


def test_dataset_get_filtered_dataset_with_single_nonpartition_success(
    timdex_dataset_multi_source,
):
    filtered_timdex_dataset = timdex_dataset_multi_source._get_filtered_dataset(
        run_id="abc123",
    )
    filtered_local_df = filtered_timdex_dataset.to_table().to_pandas()

    # timdex_dataset_multi_source consists of single 'run_id' value
    # therefore, filtered_timdex_dataset includes all records
    assert len(filtered_local_df) == filtered_timdex_dataset.count_rows()
    assert filtered_local_df["run_id"].unique() == ["abc123"]


def test_dataset_get_filtered_dataset_with_multi_nonpartition_filters_success(
    timdex_dataset_multi_source,
):
    filtered_timdex_dataset = timdex_dataset_multi_source._get_filtered_dataset(
        timdex_record_id="alma:0",
        source="alma",
        run_type="daily",
        run_id="abc123",
        action="index",
    )
    filtered_local_df = filtered_timdex_dataset.to_table().to_pandas()

    assert len(filtered_local_df) == 1
    assert filtered_local_df["timdex_record_id"].iloc[0] == "alma:0"


def test_dataset_get_filtered_dataset_with_or_nonpartition_filters_success(
    timdex_dataset_multi_source,
):
    filtered_timdex_dataset = timdex_dataset_multi_source._get_filtered_dataset(
        timdex_record_id=["alma:0", "alma:1"]
    )
    filtered_local_df = filtered_timdex_dataset.to_table().to_pandas()
    assert len(filtered_local_df) == 2
    assert filtered_local_df["timdex_record_id"].tolist() == ["alma:0", "alma:1"]


def test_dataset_get_filtered_dataset_with_run_date_str_successs(
    timdex_dataset_multi_source,
):
    filtered_timdex_dataset = timdex_dataset_multi_source._get_filtered_dataset(
        run_date="2024-12-01"
    )
    empty_timdex_dataset = timdex_dataset_multi_source._get_filtered_dataset(
        run_date="2024-12-02"
    )

    # timdex_dataset_multi_source consists of single 'run_date' value
    # therefore, filtered_timdex_dataset includes all records
    assert (
        filtered_timdex_dataset.count_rows()
        == timdex_dataset_multi_source.dataset.count_rows()
    )
    assert empty_timdex_dataset.count_rows() == 0


def test_dataset_get_filtered_dataset_with_run_date_obj_success(
    timdex_dataset_multi_source,
):
    filtered_timdex_dataset = timdex_dataset_multi_source._get_filtered_dataset(
        run_date=date(2024, 12, 1)
    )
    empty_timdex_dataset = timdex_dataset_multi_source._get_filtered_dataset(
        run_date=date(2024, 12, 2)
    )

    # timdex_dataset_multi_source consists of single 'run_date' value
    # therefore, filtered_timdex_dataset includes all records
    assert (
        filtered_timdex_dataset.count_rows()
        == timdex_dataset_multi_source.dataset.count_rows()
    )
    assert empty_timdex_dataset.count_rows() == 0


def test_dataset_get_filtered_dataset_with_ymd_success(timdex_dataset_multi_source):
    filtered_timdex_dataset = timdex_dataset_multi_source._get_filtered_dataset(
        year="2024"
    )
    empty_timdex_dataset = timdex_dataset_multi_source._get_filtered_dataset(year="2025")

    # timdex_dataset_multi_source consists of single 'run_date' value
    # therefore, filtered_timdex_dataset includes all records
    assert (
        filtered_timdex_dataset.count_rows()
        == timdex_dataset_multi_source.dataset.count_rows()
    )
    assert empty_timdex_dataset.count_rows() == 0


def test_dataset_get_filtered_dataset_with_run_date_invalid_raise_error(
    timdex_dataset_multi_source,
):
    with pytest.raises(
        TypeError,
        match=(
            "Provided 'run_date' value must be a string matching format '%Y-%m-%d' "
            "or a datetime.date."
        ),
    ):
        _ = timdex_dataset_multi_source._get_filtered_dataset(run_date=999)


def test_dataset_get_s3_filesystem_success(mocker):
    mocked_s3_filesystem = mocker.spy(fs, "S3FileSystem")
    s3_filesystem = TIMDEXDataset.get_s3_filesystem()

    assert mocked_s3_filesystem.call_args[1] == {
        "secret_key": "fake_secret_key",
        "access_key": "fake_access_key",
        "region": "us-east-1",
        "session_token": "fake_session_token",
    }
    assert isinstance(s3_filesystem, pa._s3fs.S3FileSystem)


def test_dataset_timdex_dataset_validate_success(timdex_dataset):
    assert timdex_dataset.dataset.to_table().validate() is None  # where None is valid


def test_dataset_timdex_dataset_row_count_success(timdex_dataset):
    assert timdex_dataset.dataset.count_rows() == timdex_dataset.dataset.count_rows()


def test_dataset_all_records_not_current_and_not_deduped(timdex_dataset_with_runs):
    all_records_df = timdex_dataset_with_runs.read_dataframe()

    # assert counts reflect all records from dataset, no deduping
    assert all_records_df.source.value_counts().to_dict() == {"alma": 254, "dspace": 194}

    # assert run_date min/max dates align with min/max for all runs
    assert all_records_df.run_date.min() == date(2024, 12, 1)
    assert all_records_df.run_date.max() == date(2025, 2, 5)


def test_dataset_records_data_structure_is_idempotent(timdex_dataset_with_runs):
    assert os.path.exists(timdex_dataset_with_runs.data_records_root)
    start_file_count = glob.glob(f"{timdex_dataset_with_runs.data_records_root}/**/*")

    timdex_dataset_with_runs.create_data_structure()

    assert os.path.exists(timdex_dataset_with_runs.data_records_root)
    end_file_count = glob.glob(f"{timdex_dataset_with_runs.data_records_root}/**/*")
    assert start_file_count == end_file_count
