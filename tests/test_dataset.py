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
    DatasetNotLoadedError,
    TIMDEXDataset,
    TIMDEXDatasetConfig,
)


@pytest.mark.parametrize(
    ("location_param", "expected_file_system", "expected_source_param"),
    [
        (
            "path/to/dataset",
            fs.LocalFileSystem,
            "path/to/dataset/data/records",
        ),
        (
            "s3://timdex/path/to/dataset",
            fs.S3FileSystem,
            "timdex/path/to/dataset/data/records",
        ),
    ],
)
def test_dataset_init_success(
    location_param,
    expected_file_system,
    expected_source_param,
    s3_bucket_mocked,
    tmp_path,
):
    location = location_param
    expected_source = expected_source_param

    if not location.startswith("s3://"):
        location = str(tmp_path / location)
        expected_source = str(tmp_path / expected_source)

    timdex_dataset = TIMDEXDataset(location=location)
    assert isinstance(timdex_dataset.filesystem, expected_file_system)
    assert timdex_dataset.paths == expected_source


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
def test_dataset_load_local_sets_filesystem_and_dataset_success(
    mock_pyarrow_ds, mock_local_fs, tmp_path
):
    mock_local_fs.return_value = MagicMock()
    mock_pyarrow_ds.return_value = MagicMock()

    location = str(Path(tmp_path) / "local/path/to/dataset")

    timdex_dataset = TIMDEXDataset(location=location)
    result = timdex_dataset.load()

    mock_pyarrow_ds.assert_called_once_with(
        f"{location}/data/records",
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
    mock_pyarrow_ds, mock_get_s3_fs, s3_bucket_mocked
):
    mock_get_s3_fs.return_value = MagicMock()
    mock_pyarrow_ds.return_value = MagicMock()

    timdex_dataset = TIMDEXDataset(location="s3://timdex/path/to/dataset")
    result = timdex_dataset.load()

    mock_pyarrow_ds.assert_called_with(
        "timdex/path/to/dataset/data/records",
        schema=timdex_dataset.schema,
        format="parquet",
        partitioning="hive",
        filesystem=mock_get_s3_fs.return_value,
    )
    assert timdex_dataset.dataset == mock_pyarrow_ds.return_value
    assert result is None


def test_dataset_load_without_filters_success(timdex_dataset_multi_source):
    timdex_dataset_multi_source.load()

    assert os.path.exists(timdex_dataset_multi_source.location)
    assert timdex_dataset_multi_source.row_count == 5_000


def test_dataset_load_with_run_date_str_filters_success(timdex_dataset_multi_source):
    timdex_dataset_multi_source.load(run_date="2024-12-01")

    assert os.path.exists(timdex_dataset_multi_source.location)
    assert timdex_dataset_multi_source.row_count == 5_000


def test_dataset_load_with_run_date_obj_filters_success(timdex_dataset_multi_source):
    timdex_dataset_multi_source.load(run_date=date(2024, 12, 1))

    assert os.path.exists(timdex_dataset_multi_source.location)
    assert timdex_dataset_multi_source.row_count == 5_000


def test_dataset_load_with_ymd_filters_success(timdex_dataset_multi_source):
    timdex_dataset_multi_source.load(year="2024", month="12", day="01")

    assert os.path.exists(timdex_dataset_multi_source.location)
    assert timdex_dataset_multi_source.row_count == 5_000


def test_dataset_load_with_single_nonpartition_filters_success(
    timdex_dataset_multi_source,
):
    timdex_dataset_multi_source.load(timdex_record_id="alma:0")

    assert timdex_dataset_multi_source.row_count == 1


def test_dataset_load_with_multi_nonpartition_filters_success(
    timdex_dataset_multi_source,
):
    timdex_dataset_multi_source.load(
        timdex_record_id="alma:0",
        source="alma",
        run_type="daily",
        run_id="abc123",
        action="index",
    )

    assert timdex_dataset_multi_source.row_count == 1


@pytest.mark.skip(reason="All tests for 'current' records will be reworked.")
def test_dataset_load_current_records_all_sources_success(
    timdex_timdex_dataset_with_runs,
):
    timdex_dataset = TIMDEXDataset(timdex_timdex_dataset_with_runs.location)

    # 16 total parquet files, with current_records=False we get them all
    timdex_dataset.load(current_records=False)
    assert len(timdex_dataset.dataset.files) == 16

    # 16 total parquet files, with current_records=True we only get 12 for current runs
    timdex_dataset.load(current_records=True)
    assert len(timdex_dataset.dataset.files) == 12


@pytest.mark.skip(reason="All tests for 'current' records will be reworked.")
def test_dataset_load_current_records_one_source_success(timdex_timdex_dataset_with_runs):
    timdex_dataset = TIMDEXDataset(timdex_timdex_dataset_with_runs.location)
    timdex_dataset.load(current_records=True, source="alma")

    # 7 total parquet files for source, only 6 related to current runs
    assert len(timdex_dataset.dataset.files) == 6


def test_dataset_get_filtered_dataset_with_single_nonpartition_success(
    timdex_dataset_multi_source,
):
    timdex_dataset_multi_source.load()  # initial load dataset, no filters passed

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
    timdex_dataset_multi_source.load()  # initial load dataset, no filters passed

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
    timdex_dataset_multi_source.load()

    filtered_timdex_dataset = timdex_dataset_multi_source._get_filtered_dataset(
        timdex_record_id=["alma:0", "alma:1"]
    )
    filtered_local_df = filtered_timdex_dataset.to_table().to_pandas()
    assert len(filtered_local_df) == 2
    assert filtered_local_df["timdex_record_id"].tolist() == ["alma:0", "alma:1"]


def test_dataset_get_filtered_dataset_with_run_date_str_successs(
    timdex_dataset_multi_source,
):
    timdex_dataset_multi_source.load()  # initial load dataset, no filters passed

    filtered_timdex_dataset = timdex_dataset_multi_source._get_filtered_dataset(
        run_date="2024-12-01"
    )
    empty_timdex_dataset = timdex_dataset_multi_source._get_filtered_dataset(
        run_date="2024-12-02"
    )

    # timdex_dataset_multi_source consists of single 'run_date' value
    # therefore, filtered_timdex_dataset includes all records
    assert filtered_timdex_dataset.count_rows() == timdex_dataset_multi_source.row_count
    assert empty_timdex_dataset.count_rows() == 0


def test_dataset_get_filtered_dataset_with_run_date_obj_success(
    timdex_dataset_multi_source,
):
    timdex_dataset_multi_source.load()  # initial load dataset, no filters passed

    filtered_timdex_dataset = timdex_dataset_multi_source._get_filtered_dataset(
        run_date=date(2024, 12, 1)
    )
    empty_timdex_dataset = timdex_dataset_multi_source._get_filtered_dataset(
        run_date=date(2024, 12, 2)
    )

    # timdex_dataset_multi_source consists of single 'run_date' value
    # therefore, filtered_timdex_dataset includes all records
    assert filtered_timdex_dataset.count_rows() == timdex_dataset_multi_source.row_count
    assert empty_timdex_dataset.count_rows() == 0


def test_dataset_get_filtered_dataset_with_ymd_success(timdex_dataset_multi_source):
    timdex_dataset_multi_source.load()  # initial load dataset, no filters passed

    filtered_timdex_dataset = timdex_dataset_multi_source._get_filtered_dataset(
        year="2024"
    )
    empty_timdex_dataset = timdex_dataset_multi_source._get_filtered_dataset(year="2025")

    # timdex_dataset_multi_source consists of single 'run_date' value
    # therefore, filtered_timdex_dataset includes all records
    assert filtered_timdex_dataset.count_rows() == timdex_dataset_multi_source.row_count
    assert empty_timdex_dataset.count_rows() == 0


def test_dataset_get_filtered_dataset_with_run_date_invalid_raise_error(
    timdex_dataset_multi_source,
):
    timdex_dataset_multi_source.load()  # initial load dataset, no filters passed

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


@pytest.mark.parametrize(
    ("location_param", "expected_filesystem", "expected_source_param"),
    [
        ("path/to/dataset", fs.LocalFileSystem, "path/to/dataset"),
        (
            ["path/to/records1.parquet", "path/to/records2.parquet"],
            fs.LocalFileSystem,
            ["path/to/records1.parquet", "path/to/records2.parquet"],
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
    location_param,
    expected_filesystem,
    expected_source_param,
    tmp_path,
):
    get_s3_filesystem.return_value = fs.S3FileSystem()

    location = location_param
    expected_source = expected_source_param

    if isinstance(location, str) and not location.startswith("s3://"):
        location = str(tmp_path / location)
        expected_source = str(tmp_path / expected_source)
    elif isinstance(location, list) and not location[0].startswith("s3://"):
        location = [str(tmp_path / path) for path in location]
        expected_source = [str(tmp_path / path) for path in expected_source]

    filesystem, source = TIMDEXDataset.parse_location(location)
    assert isinstance(filesystem, expected_filesystem)
    assert source == expected_source


@pytest.mark.parametrize(
    ("location_param", "expected_exception"),
    [
        # None is invalid location type
        (None, TypeError),
        # mixed local and S3 locations
        (
            [
                "local/path/to/dataset/records.parquet",
                "s3://path/to/dataset/records.parquet",
            ],
            ValueError,
        ),
    ],
)
@patch("timdex_dataset_api.dataset.TIMDEXDataset.get_s3_filesystem")
def test_dataset_parse_location_error(
    get_s3_filesystem, location_param, expected_exception, tmp_path
):
    get_s3_filesystem.return_value = fs.S3FileSystem()

    location = location_param
    if isinstance(location, list) and not all(
        path.startswith("s3://") for path in location
    ):
        # Update the local path with tmp_path
        location = [
            str(tmp_path / path) if not path.startswith("s3://") else path
            for path in location
        ]

    with pytest.raises(expected_exception):
        _ = TIMDEXDataset.parse_location(location)


def test_dataset_timdex_dataset_validate_success(timdex_dataset):
    assert timdex_dataset.dataset.to_table().validate() is None  # where None is valid


def test_dataset_timdex_dataset_row_count_success(timdex_dataset):
    assert timdex_dataset.dataset.count_rows() == timdex_dataset.row_count


def test_dataset_timdex_dataset_row_count_missing_dataset_raise_error(
    timdex_dataset, tmp_path
):
    td = TIMDEXDataset(location=str(tmp_path / "path/to/nowhere"))
    with pytest.raises(DatasetNotLoadedError):
        _ = td.row_count


def test_dataset_all_records_not_current_and_not_deduped(timdex_dataset_with_runs):
    timdex_dataset_with_runs.load()
    all_records_df = timdex_dataset_with_runs.read_dataframe()

    # assert counts reflect all records from dataset, no deduping
    assert all_records_df.source.value_counts().to_dict() == {"alma": 254, "dspace": 194}

    # assert run_date min/max dates align with min/max for all runs
    assert all_records_df.run_date.min() == date(2024, 12, 1)
    assert all_records_df.run_date.max() == date(2025, 2, 5)


@pytest.mark.skip(reason="All tests for 'current' records will be reworked.")
def test_dataset_all_current_records_deduped(timdex_dataset_with_runs):
    timdex_dataset_with_runs.load(current_records=True)
    all_records_df = timdex_dataset_with_runs.read_dataframe()

    # assert both sources have accurate record counts for current records only
    assert all_records_df.source.value_counts().to_dict() == {"dspace": 90, "alma": 100}

    # assert only one "full" run, per source
    assert len(all_records_df[all_records_df.run_type == "full"].run_id.unique()) == 2

    # assert run_date min/max dates align with both sources min/max dates
    assert all_records_df.run_date.min() == date(2025, 1, 1)  # both
    assert all_records_df.run_date.max() == date(2025, 2, 5)  # dspace


@pytest.mark.skip(reason="All tests for 'current' records will be reworked.")
def test_dataset_source_current_records_deduped(timdex_dataset_with_runs):
    timdex_dataset_with_runs.load(current_records=True, source="alma")
    alma_records_df = timdex_dataset_with_runs.read_dataframe()

    # assert only alma records present and correct count
    assert alma_records_df.source.value_counts().to_dict() == {"alma": 100}

    # assert only one "full" run
    assert len(alma_records_df[alma_records_df.run_type == "full"].run_id.unique()) == 1

    # assert run_date min/max dates are correct for single source
    assert alma_records_df.run_date.min() == date(2025, 1, 1)
    assert alma_records_df.run_date.max() == date(2025, 1, 5)


@pytest.mark.skip(reason="All tests for 'current' records will be reworked.")
def test_dataset_all_read_methods_get_deduplication(
    timdex_dataset_with_runs,
):
    timdex_dataset_with_runs.load(current_records=True, source="alma")

    full_df = timdex_dataset_with_runs.read_dataframe()
    all_records = list(timdex_dataset_with_runs.read_dicts_iter())
    transformed_records = list(timdex_dataset_with_runs.read_transformed_records_iter())

    assert len(full_df) == len(all_records) == len(transformed_records)


@pytest.mark.skip(reason="All tests for 'current' records will be reworked.")
def test_dataset_current_records_no_additional_filtering_accurate_records_yielded(
    timdex_dataset_with_runs,
):
    timdex_dataset_with_runs.load(current_records=True, source="alma")
    df = timdex_dataset_with_runs.read_dataframe()
    assert df.action.value_counts().to_dict() == {"index": 99, "delete": 1}


@pytest.mark.skip(reason="All tests for 'current' records will be reworked.")
def test_dataset_current_records_action_filtering_accurate_records_yielded(
    timdex_dataset_with_runs,
):
    timdex_dataset_with_runs.load(current_records=True, source="alma")
    df = timdex_dataset_with_runs.read_dataframe(action="index")
    assert df.action.value_counts().to_dict() == {"index": 99}


@pytest.mark.skip(reason="All tests for 'current' records will be reworked.")
def test_dataset_current_records_index_filtering_accurate_records_yielded(
    timdex_dataset_with_runs,
):
    """This is a somewhat complex test, but demonstrates that only 'current' records
    are yielded when .load(current_records=True) is applied.

    Given these runs from the fixture:
    [
        ...
        (25, "alma", "2025-01-03", "daily", "index", "run-5"),   <---- filtered to
        (10, "alma", "2025-01-04", "daily", "delete", "run-6"),  <---- influences current
        ...
    ]

    Though we are filtering to run-5, which has 25 total records to-index, we see only 15
    records yielded.  Why?  This is because while we have filtered to only yield from
    run-5, run-6 had 10 deletes which made records alma:0|9 no longer "current" in run-5.
    As we yielded records reverse chronologically, the deletes from run-6 (alma:0-alma:9)
    "influenced" what records we would see as we continue backwards in time.
    """
    # with current_records=False, we get all 25 records from run-5
    timdex_dataset_with_runs.load(current_records=False, source="alma")
    df = timdex_dataset_with_runs.read_dataframe(run_id="run-5")
    assert len(df) == 25

    # with current_records=True, we only get 15 records from run-5
    # because newer run-6 influenced what records are current for older run-5
    timdex_dataset_with_runs.load(current_records=True, source="alma")
    df = timdex_dataset_with_runs.read_dataframe(run_id="run-5")
    assert len(df) == 15
    assert list(df.timdex_record_id) == [
        "alma:10",
        "alma:11",
        "alma:12",
        "alma:13",
        "alma:14",
        "alma:15",
        "alma:16",
        "alma:17",
        "alma:18",
        "alma:19",
        "alma:20",
        "alma:21",
        "alma:22",
        "alma:23",
        "alma:24",
    ]


@pytest.mark.skip(reason="All tests for 'current' records will be reworked.")
def test_dataset_load_current_records_gets_correct_same_day_full_run(
    timdex_dataset_same_day_runs,
):
    """Two full runs were performed on the same day, but 'run-2' was performed most
    recently.  current_records=True should discover the more recent of the two 'run-2',
    not 'run-1'."""
    timdex_dataset_same_day_runs.load(current_records=True, run_type="full")
    df = timdex_dataset_same_day_runs.read_dataframe()

    assert list(df.run_id.unique()) == ["run-2"]


@pytest.mark.skip(reason="All tests for 'current' records will be reworked.")
def test_dataset_load_current_records_gets_correct_same_day_daily_runs_ordering(
    timdex_dataset_same_day_runs,
):
    """Two runs were performed on 2025-01-02, but the most recent records should be from
    run 'run-5' which are action='delete', not 'run-4' with action='index'."""
    timdex_dataset_same_day_runs.load(current_records=True, run_type="daily")
    first_record = next(timdex_dataset_same_day_runs.read_dicts_iter())

    assert first_record["run_id"] == "run-5"
    assert first_record["action"] == "delete"


def test_dataset_records_data_structure_is_idempotent(timdex_dataset_with_runs):
    assert os.path.exists(timdex_dataset_with_runs.data_records_root)
    start_file_count = glob.glob(f"{timdex_dataset_with_runs.data_records_root}/**/*")

    timdex_dataset_with_runs.create_data_structure()

    assert os.path.exists(timdex_dataset_with_runs.data_records_root)
    end_file_count = glob.glob(f"{timdex_dataset_with_runs.data_records_root}/**/*")
    assert start_file_count == end_file_count
