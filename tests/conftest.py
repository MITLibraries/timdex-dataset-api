"""tests/conftest.py"""

from collections.abc import Iterator

import boto3
import moto
import pytest

from tests.utils import generate_sample_records
from timdex_dataset_api import TIMDEXDataset, TIMDEXDatasetMetadata
from timdex_dataset_api.dataset import TIMDEXDatasetConfig
from timdex_dataset_api.record import DatasetRecord


@pytest.fixture(autouse=True)
def _test_env(monkeypatch):
    monkeypatch.setenv("TDA_LOG_LEVEL", "INFO")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "fake_access_key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fake_secret_key")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "fake_session_token")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.delenv("MINIO_S3_ENDPOINT_URL", raising=False)


# ================================================================================
# S3/AWS Fixtures
# ================================================================================


@pytest.fixture
def s3_bucket_name():
    """S3 bucket name for testing."""
    return "timdex"


@pytest.fixture
def s3_bucket_mocked(s3_bucket_name):
    """Mocked S3 bucket using moto."""
    with moto.mock_aws():
        conn = boto3.resource("s3", region_name="us-east-1")
        conn.create_bucket(Bucket=s3_bucket_name)
        yield conn


# ================================================================================
# Base Dataset Fixtures
# ================================================================================


@pytest.fixture
def timdex_dataset_empty(tmp_path) -> TIMDEXDataset:
    """Empty TIMDEXDataset instance without any data."""
    return TIMDEXDataset(str(tmp_path / "empty_dataset/"))


@pytest.fixture
def timdex_dataset_config() -> TIMDEXDatasetConfig:
    """Default dataset configuration that can be overridden."""
    return TIMDEXDatasetConfig()


@pytest.fixture(scope="module")
def timdex_dataset_config_small() -> TIMDEXDatasetConfig:
    """Small file configuration for testing partitioning behavior."""
    return TIMDEXDatasetConfig(max_rows_per_group=75, max_rows_per_file=75)


@pytest.fixture
def timdex_dataset(tmp_path, timdex_dataset_config) -> TIMDEXDataset:
    """Basic TIMDEXDataset with 1000 sample records from alma source."""
    dataset = TIMDEXDataset(
        str(tmp_path / "basic_dataset/"), config=timdex_dataset_config
    )
    dataset.write(
        generate_sample_records(
            num_records=1000,
            source="alma",
            run_date="2024-12-01",
            run_type="full",
            action="index",
            run_id="test-run-1",
        ),
        write_append_deltas=False,
    )
    return dataset


@pytest.fixture(scope="module")
def timdex_dataset_multi_source(tmp_path_factory) -> TIMDEXDataset:
    """TIMDEXDataset with multiple sources for testing filtering.

    Contains 1000 records each from: alma, dspace, aspace, libguides, gismit
    """
    dataset_dir = tmp_path_factory.mktemp("multi_source_dataset_mod")
    dataset = TIMDEXDataset(str(dataset_dir))

    for source, run_id in [
        ("alma", "abc123"),
        ("dspace", "def456"),
        ("aspace", "ghi789"),
        ("libguides", "jkl123"),
        ("gismit", "mno456"),
    ]:
        dataset.write(
            generate_sample_records(
                num_records=1000,
                source=source,
                run_date="2024-12-01",
                run_id=run_id,
            ),
            write_append_deltas=False,
        )

    # ensure static metadata database exists for read methods
    dataset.metadata.recreate_static_database_file()
    dataset.metadata.refresh()

    return dataset


@pytest.fixture(scope="module")
def timdex_dataset_with_runs(
    tmp_path_factory, timdex_dataset_config_small
) -> TIMDEXDataset:
    """TIMDEXDataset with multiple full and daily ETL runs.

    Simulates realistic ETL pattern with:
    - Multiple sources (alma, dspace)
    - Full and daily runs
    - Index and delete actions
    - Small file sizes to test partitioning
    """
    dataset = TIMDEXDataset(
        str(tmp_path_factory.mktemp("dataset_with_runs_mod")),
        config=timdex_dataset_config_small,
    )

    # alma ETL runs
    alma_runs = [
        (40, "alma", "2024-12-01", "full", "index", "run-1"),
        (20, "alma", "2024-12-15", "daily", "index", "run-2"),
        (100, "alma", "2025-01-01", "full", "index", "run-3"),
        (50, "alma", "2025-01-02", "daily", "index", "run-4"),
        (25, "alma", "2025-01-03", "daily", "index", "run-5"),
        (10, "alma", "2025-01-04", "daily", "delete", "run-6"),
        (9, "alma", "2025-01-05", "daily", "index", "run-7"),
    ]

    # dspace ETL runs
    dspace_runs = [
        (30, "dspace", "2024-12-02", "full", "index", "run-8"),
        (10, "dspace", "2024-12-16", "daily", "index", "run-9"),
        (90, "dspace", "2025-02-01", "full", "index", "run-10"),
        (40, "dspace", "2025-02-02", "daily", "index", "run-11"),
        (15, "dspace", "2025-02-03", "daily", "index", "run-12"),
        (5, "dspace", "2025-02-04", "daily", "delete", "run-13"),
        (4, "dspace", "2025-02-05", "daily", "index", "run-14"),
    ]

    for num_records, source, run_date, run_type, action, run_id in (
        alma_runs + dspace_runs
    ):
        dataset.write(
            generate_sample_records(
                num_records=num_records,
                source=source,
                run_date=run_date,
                run_type=run_type,
                action=action,
                run_id=run_id,
            ),
            write_append_deltas=False,
        )

    # We intentionally DO NOT create the static metadata here since some tests
    # expect it to be missing initially. Use a separate fixture when metadata is required.

    return dataset


@pytest.fixture
def timdex_dataset_same_day_runs(tmp_path) -> TIMDEXDataset:
    """TIMDEXDataset with multiple runs on the same day for testing run ordering.

    Tests proper handling of:
    - Multiple full runs on same day (run-2 should establish baseline)
    - Multiple daily runs on same day (deletes should be after indexes)
    - Expected result: 70 records (75 base - 5 deletes)
    """
    dataset = TIMDEXDataset(str(tmp_path / "same_day_runs_dataset/"))

    runs = [
        (100, "alma", "2025-01-01", "full", "index", "run-1", "2025-01-01T01:00:00"),
        (75, "alma", "2025-01-01", "full", "index", "run-2", "2025-01-01T02:00:00"),
        (10, "alma", "2025-01-01", "daily", "index", "run-3", "2025-01-01T03:00:00"),
        (20, "alma", "2025-01-02", "daily", "index", "run-4", "2025-01-02T01:00:00"),
        (5, "alma", "2025-01-02", "daily", "delete", "run-5", "2025-01-02T02:00:00"),
    ]

    for num_records, source, run_date, run_type, action, run_id, run_timestamp in runs:
        dataset.write(
            generate_sample_records(
                num_records=num_records,
                source=source,
                run_date=run_date,
                run_type=run_type,
                action=action,
                run_id=run_id,
                run_timestamp=run_timestamp,
            ),
            write_append_deltas=False,
        )
    return dataset


# ================================================================================
# Dataset Metadata Fixtures
# ================================================================================


@pytest.fixture(scope="module")
def timdex_metadata(timdex_dataset_with_runs) -> TIMDEXDatasetMetadata:
    """TIMDEXDatasetMetadata with static database file created."""
    metadata = TIMDEXDatasetMetadata(timdex_dataset_with_runs.location)
    metadata.recreate_static_database_file()
    metadata.refresh()
    return metadata


@pytest.fixture(scope="module")
def timdex_dataset_with_runs_with_metadata(
    timdex_dataset_with_runs,
) -> TIMDEXDataset:
    """TIMDEXDataset with runs and static metadata created for read tests."""
    timdex_dataset_with_runs.metadata.recreate_static_database_file()
    timdex_dataset_with_runs.metadata.refresh()
    return timdex_dataset_with_runs


@pytest.fixture
def timdex_metadata_empty(timdex_dataset_with_runs) -> TIMDEXDatasetMetadata:
    """TIMDEXDatasetMetadata without static database file."""
    return TIMDEXDatasetMetadata(timdex_dataset_with_runs.location)


@pytest.fixture
def timdex_metadata_with_deltas(
    timdex_dataset_with_runs, timdex_metadata
) -> TIMDEXDatasetMetadata:
    """TIMDEXDatasetMetadata with append deltas from additional writes."""
    td = TIMDEXDataset(timdex_dataset_with_runs.location)

    # perform an ETL write of 50 records
    # results in 1 append delta with 50 rows therein
    records = generate_sample_records(
        num_records=50,
        source="alma",
        run_date="2025-01-10",
        run_type="daily",
        action="index",
        run_id="run-delta-1",
    )
    td.write(records)

    return TIMDEXDatasetMetadata(timdex_dataset_with_runs.location)


# ================================================================================
# Utility Fixtures
# ================================================================================


@pytest.fixture
def sample_records() -> Iterator[DatasetRecord]:
    """Generate 100 sample records with default parameters."""
    return generate_sample_records(num_records=100)


@pytest.fixture
def sample_records_generator():
    """Factory fixture for generating custom sample records."""

    def _generate(num_records: int = 100, **kwargs) -> Iterator[DatasetRecord]:
        return generate_sample_records(num_records=num_records, **kwargs)

    return _generate
