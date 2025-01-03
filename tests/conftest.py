"""tests/conftest.py"""

# ruff: noqa: D205, D209


import pytest

from tests.utils import (
    generate_sample_records,
    generate_sample_records_with_simulated_partitions,
)
from timdex_dataset_api import TIMDEXDataset


@pytest.fixture(autouse=True)
def _test_env(monkeypatch):
    monkeypatch.setenv("TDA_LOG_LEVEL", "INFO")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "fake_access_key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fake_secret_key")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "fake_session_token")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


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
