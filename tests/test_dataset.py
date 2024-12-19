# ruff: noqa: S105, S106, SLF001
import os
from datetime import date
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pytest
from pyarrow import fs

from timdex_dataset_api.dataset import DatasetNotLoadedError, TIMDEXDataset


@pytest.mark.parametrize(
    ("location", "expected_file_system", "expected_source"),
    [
        ("path/to/dataset", fs.LocalFileSystem, "path/to/dataset"),
        ("s3://bucket/path/to/dataset", fs.S3FileSystem, "bucket/path/to/dataset"),
    ],
)
def test_dataset_init_success(location, expected_file_system, expected_source):
    timdex_dataset = TIMDEXDataset(location=location)
    assert isinstance(timdex_dataset.filesystem, expected_file_system)
    assert timdex_dataset.source == expected_source


@patch("timdex_dataset_api.dataset.fs.LocalFileSystem")
@patch("timdex_dataset_api.dataset.ds.dataset")
def test_dataset_load_local_sets_filesystem_and_dataset_success(
    mock_pyarrow_ds, mock_local_fs
):
    mock_local_fs.return_value = MagicMock()
    mock_pyarrow_ds.return_value = MagicMock()

    timdex_dataset = TIMDEXDataset(location="local/path/to/dataset")
    result = timdex_dataset.load()

    mock_pyarrow_ds.assert_called_once_with(
        "local/path/to/dataset/",
        schema=timdex_dataset.schema,
        format="parquet",
        partitioning="hive",
        filesystem=mock_local_fs.return_value,
    )

    assert timdex_dataset.dataset == mock_pyarrow_ds.return_value
    assert result is None


@patch("timdex_dataset_api.dataset.TIMDEXDataset.get_s3_filesystem")
@patch("timdex_dataset_api.dataset.ds.dataset")
def test_dataset_load_s3_sets_filesystem_and_dataset_success(
    mock_pyarrow_ds, mock_get_s3_fs
):
    mock_get_s3_fs.return_value = MagicMock()
    mock_pyarrow_ds.return_value = MagicMock()

    timdex_dataset = TIMDEXDataset(location="s3://bucket/path/to/dataset")
    result = timdex_dataset.load()

    mock_get_s3_fs.assert_called_once()
    mock_pyarrow_ds.assert_called_once_with(
        "bucket/path/to/dataset/",
        schema=timdex_dataset.schema,
        format="parquet",
        partitioning="hive",
        filesystem=mock_get_s3_fs.return_value,
    )
    assert timdex_dataset.dataset == mock_pyarrow_ds.return_value
    assert result is None


@patch("timdex_dataset_api.dataset.fs.LocalFileSystem")
@patch("timdex_dataset_api.dataset.ds.dataset")
def test_dataset_load_with_partition_prefix_via_run_date_success(
    mock_pyarrow_ds, mock_local_fs
):
    mock_local_fs.return_value = MagicMock()
    mock_pyarrow_ds.return_value = MagicMock()

    timdex_dataset = TIMDEXDataset(location="local/path/to/dataset")
    timdex_dataset.load(run_date="2024-12-01")

    mock_pyarrow_ds.assert_called_once_with(
        "local/path/to/dataset/year=2024/month=12/day=01",
        schema=timdex_dataset.schema,
        format="parquet",
        partitioning="hive",
        filesystem=mock_local_fs.return_value,
    )


@patch("timdex_dataset_api.dataset.fs.LocalFileSystem")
@patch("timdex_dataset_api.dataset.ds.dataset")
def test_dataset_load_with_partition_prefix_via_run_date_components_success(
    mock_pyarrow_ds, mock_local_fs
):
    mock_local_fs.return_value = MagicMock()
    mock_pyarrow_ds.return_value = MagicMock()

    timdex_dataset = TIMDEXDataset(location="local/path/to/dataset")
    timdex_dataset.load(year="2024")

    mock_pyarrow_ds.assert_called_once_with(
        "local/path/to/dataset/year=2024",
        schema=timdex_dataset.schema,
        format="parquet",
        partitioning="hive",
        filesystem=mock_local_fs.return_value,
    )


def test_dataset_load_no_filters_success(fixed_local_dataset):
    fixed_local_dataset.load()

    assert os.path.exists(fixed_local_dataset.location)
    assert fixed_local_dataset.row_count == 5_000  # noqa: PLR2004


def test_dataset_load_and_filter_by_non_partition_field_success(fixed_local_dataset):
    fixed_local_dataset.load(timdex_record_id="alma:0")

    assert fixed_local_dataset.row_count == 1


def test_dataset_get_filtered_dataset_by_all_fields_success(fixed_local_dataset):
    fixed_local_dataset.load()  # initial load dataset, no filters passed

    filtered_local_dataset = fixed_local_dataset._get_filtered_dataset(
        timdex_record_id="alma:0",
        source="alma",
        run_type="daily",
        run_id="abc123",
        action="index",
    )
    filtered_local_df = filtered_local_dataset.to_table().to_pandas()

    assert len(filtered_local_df) == 1
    assert filtered_local_df["timdex_record_id"].iloc[0] == "alma:0"


def test_dataset_get_filtered_dataset_by_single_fields_success(fixed_local_dataset):
    fixed_local_dataset.load()  # initial load dataset, no filters passed

    filtered_local_dataset = fixed_local_dataset._get_filtered_dataset(
        run_id="abc123",
    )
    filtered_local_df = filtered_local_dataset.to_table().to_pandas()

    # fixed_local_dataset consists of single 'run_id' value
    # therefore, filtered_local_dataset includes all records
    assert len(filtered_local_df) == filtered_local_dataset.count_rows()
    assert filtered_local_df["run_id"].unique() == ["abc123"]


def test_dataset_get_filtered_dataset_by_run_date_str_successs(fixed_local_dataset):
    fixed_local_dataset.load()  # initial load dataset, no filters passed

    filtered_local_dataset = fixed_local_dataset._get_filtered_dataset(
        run_date="2024-12-01"
    )
    empty_local_dataset = fixed_local_dataset._get_filtered_dataset(run_date="2024-12-02")

    # fixed_local_dataset consists of single 'run_date' value
    # therefore, filtered_local_dataset includes all records
    assert filtered_local_dataset.count_rows() == fixed_local_dataset.row_count
    assert empty_local_dataset.count_rows() == 0


def test_dataset_get_filtered_dataset_by_run_date_date_success(fixed_local_dataset):
    fixed_local_dataset.load()  # initial load dataset, no filters passed

    filtered_local_dataset = fixed_local_dataset._get_filtered_dataset(
        run_date=date(2024, 12, 1)
    )
    empty_local_dataset = fixed_local_dataset._get_filtered_dataset(
        run_date=date(2024, 12, 2)
    )

    # fixed_local_dataset consists of single 'run_date' value
    # therefore, filtered_local_dataset includes all records
    assert filtered_local_dataset.count_rows() == fixed_local_dataset.row_count
    assert empty_local_dataset.count_rows() == 0


def test_dataset_get_filtered_dataset_by_run_date_components_success(fixed_local_dataset):
    fixed_local_dataset.load()  # initial load dataset, no filters passed

    filtered_local_dataset = fixed_local_dataset._get_filtered_dataset(year="2024")
    empty_local_dataset = fixed_local_dataset._get_filtered_dataset(year="2025")

    # fixed_local_dataset consists of single 'run_date' value
    # therefore, filtered_local_dataset includes all records
    assert filtered_local_dataset.count_rows() == fixed_local_dataset.row_count
    assert empty_local_dataset.count_rows() == 0


def test_dataset_get_filtered_dataset_by_run_date_if_invalid_type_raise_error(
    fixed_local_dataset,
):
    fixed_local_dataset.load()  # initial load dataset, no filters passed

    with pytest.raises(
        ValueError,
        match=(
            "Provided 'run_date' value must be a string matching format '%Y-%m-%d' "
            "or a datetime.date."
        ),
    ):
        _ = fixed_local_dataset._get_filtered_dataset(run_date=999)


def test_dataset_get_partition_prefixes_with_run_date_success():
    timdex_dataset = TIMDEXDataset(location="s3://bucket/path/to/dataset")

    assert (
        timdex_dataset._get_partition_prefixes(run_date="2024-12-01")
        == "year=2024/month=12/day=01"
    )


def test_dataset_get_partition_prefixes_without_run_date_success():
    timdex_dataset = TIMDEXDataset(location="s3://bucket/path/to/dataset")

    assert (
        timdex_dataset._get_partition_prefixes(year="2024", month="12", day="01")
    ) == "year=2024/month=12/day=01"
    assert (
        timdex_dataset._get_partition_prefixes(year="2024", month="12")
        == "year=2024/month=12"
    )
    assert timdex_dataset._get_partition_prefixes(year="2024") == "year=2024"


def test_dataset_get_partition_prefixes_without_run_date_raise_error():
    timdex_dataset = TIMDEXDataset(location="s3://bucket/path/to/dataset")
    with pytest.raises(
        ValueError, match="Insufficient arguments to construct a valid partition prefix."
    ):
        assert timdex_dataset._get_partition_prefixes(month="12", day="01")


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


@pytest.mark.parametrize(
    ("location", "expected_filesystem", "expected_source"),
    [
        ("/path/to/dataset", fs.LocalFileSystem, "/path/to/dataset"),
        (
            ["/path/to/records1.parquet", "/path/to/records2.parquet"],
            fs.LocalFileSystem,
            ["/path/to/records1.parquet", "/path/to/records2.parquet"],
        ),
        ("s3://bucket/path/to/dataset", fs.S3FileSystem, "bucket/path/to/dataset"),
        (
            [
                "s3://bucket/path/to/dataset/records1.parquet",
                "s3://bucket/path/to/dataset/records2.parquet",
            ],
            fs.S3FileSystem,
            [
                "bucket/path/to/dataset/records1.parquet",
                "bucket/path/to/dataset/records2.parquet",
            ],
        ),
    ],
)
@patch("timdex_dataset_api.dataset.TIMDEXDataset.get_s3_filesystem")
def test_dataset_parse_location_success(
    get_s3_filesystem,
    location,
    expected_filesystem,
    expected_source,
):
    get_s3_filesystem.return_value = fs.S3FileSystem()
    filesystem, source = TIMDEXDataset.parse_location(location)
    assert isinstance(filesystem, expected_filesystem)
    assert source == expected_source


@pytest.mark.parametrize(
    ("location", "expected_exception"),
    [
        # None is invalid location type
        (None, TypeError),
        # mixed local and S3 locations
        (
            [
                "/local/path/to/dataset/records.parquet",
                "s3://path/to/dataset/records.parquet",
            ],
            ValueError,
        ),
    ],
)
@patch("timdex_dataset_api.dataset.TIMDEXDataset.get_s3_filesystem")
def test_dataset_parse_location_error(get_s3_filesystem, location, expected_exception):
    get_s3_filesystem.return_value = fs.S3FileSystem()
    with pytest.raises(expected_exception):
        _ = TIMDEXDataset.parse_location(location)


def test_dataset_local_dataset_validate_success(local_dataset):
    assert local_dataset.dataset.to_table().validate() is None  # where None is valid


def test_dataset_local_dataset_row_count_success(local_dataset):
    assert local_dataset.dataset.count_rows() == local_dataset.row_count


def test_dataset_local_dataset_row_count_missing_dataset_raise_error(local_dataset):
    td = TIMDEXDataset(location="path/to/nowhere")
    with pytest.raises(DatasetNotLoadedError):
        _ = td.row_count
