# ruff: noqa: S105, S106, SLF001

from unittest.mock import MagicMock, patch

import pytest
from pyarrow import fs

from timdex_dataset_api.dataset import DatasetNotLoadedError, TIMDEXDataset


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
def test_parse_location_single_local_directory(
    get_s3_filesystem,
    location,
    expected_filesystem,
    expected_source,
):
    get_s3_filesystem.return_value = fs.S3FileSystem()
    filesystem, source = TIMDEXDataset.parse_location(location)
    assert isinstance(filesystem, expected_filesystem)
    assert source == expected_source


@patch("timdex_dataset_api.dataset.fs.S3FileSystem")
@patch("timdex_dataset_api.dataset.boto3.session.Session")
def test_get_s3_filesystem_success(mock_session, mock_s3_filesystem):
    mock_credentials = MagicMock()
    mock_credentials.secret_key = "fake_secret_key"
    mock_credentials.access_key = "fake_access_key"
    mock_credentials.token = "fake_session_token"
    mock_session.return_value.get_credentials.return_value = mock_credentials
    mock_session.return_value.region_name = "us-east-1"

    s3_filesystem = TIMDEXDataset.get_s3_filesystem()

    mock_s3_filesystem.assert_called_once_with(
        secret_key="fake_secret_key",
        access_key="fake_access_key",
        region="us-east-1",
        session_token="fake_session_token",
    )
    assert isinstance(s3_filesystem, MagicMock)


@patch("timdex_dataset_api.dataset.fs.LocalFileSystem")
@patch("timdex_dataset_api.dataset.ds.dataset")
def test_load_local_dataset_correct_filesystem_and_source(mock_pyarrow_ds, mock_local_fs):
    mock_local_fs.return_value = MagicMock()
    mock_pyarrow_ds.return_value = MagicMock()

    timdex_dataset = TIMDEXDataset(location="local/path/to/dataset")
    loaded_dataset = timdex_dataset.load_dataset()

    mock_pyarrow_ds.assert_called_once_with(
        "local/path/to/dataset",
        schema=timdex_dataset.schema,
        format="parquet",
        partitioning="hive",
        filesystem=mock_local_fs.return_value,
    )
    assert loaded_dataset == mock_pyarrow_ds.return_value


@patch("timdex_dataset_api.dataset.TIMDEXDataset.get_s3_filesystem")
@patch("timdex_dataset_api.dataset.ds.dataset")
def test_load_s3_dataset_correct_filesystem_and_source(mock_pyarrow_ds, mock_get_s3_fs):
    mock_get_s3_fs.return_value = MagicMock()
    mock_pyarrow_ds.return_value = MagicMock()

    timdex_dataset = TIMDEXDataset(location="s3://bucket/path/to/dataset")
    loaded_dataset = timdex_dataset.load_dataset()

    mock_get_s3_fs.assert_called_once()
    mock_pyarrow_ds.assert_called_once_with(
        "bucket/path/to/dataset",
        schema=timdex_dataset.schema,
        format="parquet",
        partitioning="hive",
        filesystem=mock_get_s3_fs.return_value,
    )
    assert loaded_dataset == mock_pyarrow_ds.return_value


@patch("timdex_dataset_api.dataset.TIMDEXDataset.load_dataset")
def test_load_method_loads_dataset_and_returns_timdexdataset_instance(mock_load_dataset):
    mock_load_dataset.return_value = MagicMock()

    timdex_dataset = TIMDEXDataset.load("s3://bucket/path/to/dataset")

    assert isinstance(timdex_dataset, TIMDEXDataset)
    assert timdex_dataset.location == "s3://bucket/path/to/dataset"
    mock_load_dataset.assert_called_once()


def test_local_dataset_is_valid(local_dataset):
    assert local_dataset.dataset.to_table().validate() is None  # where None is valid


def test_local_dataset_row_count_success(local_dataset):
    assert local_dataset.dataset.count_rows() == local_dataset.row_count


def test_local_dataset_row_count_missing_dataset_exception(local_dataset):
    td = TIMDEXDataset(location="path/to/nowhere")
    with pytest.raises(DatasetNotLoadedError):
        _ = td.row_count