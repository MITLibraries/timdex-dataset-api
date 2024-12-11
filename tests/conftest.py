"""tests/conftest.py"""

# ruff: noqa: D205, D209


import pytest

from tests.utils import generate_sample_records
from timdex_dataset_api import TIMDEXDataset


@pytest.fixture(autouse=True)
def _test_env(monkeypatch):
    monkeypatch.setenv("TDA_LOG_LEVEL", "INFO")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "fake_access_key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fake_secret_key")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "fake_session_token")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


@pytest.fixture
def local_dataset_location():
    return "tests/fixtures/local_datasets/dataset"


@pytest.fixture
def local_dataset(local_dataset_location):
    return TIMDEXDataset.load(local_dataset_location)


@pytest.fixture
def new_dataset(tmp_path) -> TIMDEXDataset:
    location = str(tmp_path / "new_dataset")
    return TIMDEXDataset(location=location)


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
