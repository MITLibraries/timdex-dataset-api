"""tests/conftest.py"""

# ruff: noqa: D205, D209

import os

import pytest

from tests.utils import (
    generate_sample_records,
    generate_sample_records_with_simulated_partitions,
)
from timdex_dataset_api import TIMDEXDataset, TIMDEXDatasetMetadata
from timdex_dataset_api.dataset import TIMDEXDatasetConfig


@pytest.fixture(autouse=True)
def _test_env(monkeypatch):
    monkeypatch.setenv("TDA_LOG_LEVEL", "INFO")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "fake_access_key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fake_secret_key")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "fake_session_token")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.delenv("MINIO_S3_ENDPOINT_URL", raising=False)


@pytest.fixture
def local_dataset_location(tmp_path):
    return str(tmp_path / "local_dataset/")


@pytest.fixture
def local_dataset(local_dataset_location):
    timdex_dataset = TIMDEXDataset(local_dataset_location)
    timdex_dataset.write(
        generate_sample_records_with_simulated_partitions(num_records=5_000)
    )
    timdex_dataset.load()
    return timdex_dataset


@pytest.fixture
def new_local_dataset(tmp_path) -> TIMDEXDataset:
    return TIMDEXDataset(location=str(tmp_path / "new_local_dataset/"))


@pytest.fixture
def fixed_local_dataset(tmp_path) -> TIMDEXDataset:
    """Local dataset with a fixed set of configurations.

    This fixture is required to perform unit tests for TIMDEXDataset.filter
    method.
    """
    timdex_dataset = TIMDEXDataset(str(tmp_path / "fixed_local_dataset/"))
    for source, run_id in [
        ("alma", "abc123"),
        ("dspace", "def456"),
        ("aspace", "ghi789"),
        ("libguides", "jkl123"),
        ("gismit", "mno456"),
    ]:
        timdex_dataset.write(
            generate_sample_records(
                num_records=1_000,
                timdex_record_id_prefix=source,
                source=source,
                run_date="2024-12-01",
                run_id=run_id,
            )
        )
    timdex_dataset.load()
    return timdex_dataset


@pytest.fixture
def sample_records_iter():
    """Simulates an iterator of X number of valid DatasetRecord instances."""

    def _records_iter(num_records):
        return generate_sample_records(num_records)

    return _records_iter


@pytest.fixture
def sample_records_iter_without_partitions():
    """Simulates an iterator of X number of DatasetRecord instances WITHOUT partition
    values included."""

    def _records_iter(num_records):
        return generate_sample_records(
            num_records, run_date="invalid run-date", year=None, month=None, day=None
        )

    return _records_iter


@pytest.fixture
def dataset_with_runs_location(tmp_path) -> str:
    """Fixture to simulate a dataset with multiple full and daily ETL runs."""
    location = str(tmp_path / "dataset_with_runs")
    os.mkdir(location)

    timdex_dataset = TIMDEXDataset(
        location, config=TIMDEXDatasetConfig(max_rows_per_group=75, max_rows_per_file=75)
    )
    timdex_dataset.load()

    run_params = []

    # simulate ETL runs for 'alma'
    run_params.extend(
        [
            (40, "alma", "2024-12-01", "full", "index", "run-1"),
            (20, "alma", "2024-12-15", "daily", "index", "run-2"),
            (100, "alma", "2025-01-01", "full", "index", "run-3"),
            (50, "alma", "2025-01-02", "daily", "index", "run-4"),
            (25, "alma", "2025-01-03", "daily", "index", "run-5"),
            (10, "alma", "2025-01-04", "daily", "delete", "run-6"),
            (9, "alma", "2025-01-05", "daily", "index", "run-7"),
        ]
    )

    # simulate ETL runs for 'dspace'
    run_params.extend(
        [
            (30, "dspace", "2024-12-02", "full", "index", "run-8"),
            (10, "dspace", "2024-12-16", "daily", "index", "run-9"),
            (90, "dspace", "2025-02-01", "full", "index", "run-10"),
            (40, "dspace", "2025-02-02", "daily", "index", "run-11"),
            (15, "dspace", "2025-02-03", "daily", "index", "run-12"),
            (5, "dspace", "2025-02-04", "daily", "delete", "run-13"),
            (4, "dspace", "2025-02-05", "daily", "index", "run-14"),
        ]
    )

    # write to dataset
    for params in run_params:
        num_records, source, run_date, run_type, action, run_id = params
        records = generate_sample_records(
            num_records,
            timdex_record_id_prefix=source,
            source=source,
            run_date=run_date,
            run_type=run_type,
            action=action,
            run_id=run_id,
        )
        timdex_dataset.write(records)

    return location


@pytest.fixture
def local_dataset_with_runs(dataset_with_runs_location) -> TIMDEXDataset:
    return TIMDEXDataset(dataset_with_runs_location)


@pytest.fixture
def dataset_with_same_day_runs(tmp_path) -> TIMDEXDataset:
    """Dataset fixture where a single source had multiple runs on the same day.

    After these runs, we'd expect 70 records in Opensearch:
        - most recent full run "run-2" established a 75 record base
        - runs "run-3" and "run-4" just modified records; no record count change
        - run "run-5" deleted 5 records

    If the order of full runs 1 & 2 are not handled correctly, we'd see an incorrect
    baseline of 100 records.

    If the order of daily runs 4 & 5 are not handled correctly, we'd see 75 records
    because the deletes would happen before the index just recreated the records.
    """
    location = str(tmp_path / "dataset_with_same_day_runs")
    os.mkdir(location)

    timdex_dataset = TIMDEXDataset(location)

    run_params = []

    # Simulate two "full" runs where "run-2" should establish the baseline.
    # Simulate daily runs, multiple per day sometimes, where deletes from "run-5" should
    # be represented.
    run_params.extend(
        [
            (100, "alma", "2025-01-01", "full", "index", "run-1", "2025-01-01T01:00:00"),
            (75, "alma", "2025-01-01", "full", "index", "run-2", "2025-01-01T02:00:00"),
            (10, "alma", "2025-01-01", "daily", "index", "run-3", "2025-01-01T03:00:00"),
            (20, "alma", "2025-01-02", "daily", "index", "run-4", "2025-01-02T01:00:00"),
            (5, "alma", "2025-01-02", "daily", "delete", "run-5", "2025-01-02T02:00:00"),
        ]
    )

    for params in run_params:
        num_records, source, run_date, run_type, action, run_id, run_timestamp = params
        records = generate_sample_records(
            num_records,
            timdex_record_id_prefix=source,
            source=source,
            run_date=run_date,
            run_type=run_type,
            action=action,
            run_id=run_id,
            run_timestamp=run_timestamp,
        )
        timdex_dataset.write(records)

    # reload after writes
    timdex_dataset.load()

    return timdex_dataset


@pytest.fixture
def timdex_dataset_metadata(dataset_with_same_day_runs):
    return TIMDEXDatasetMetadata(timdex_dataset=dataset_with_same_day_runs)
