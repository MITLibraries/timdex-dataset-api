"""tests/conftest.py"""

import pytest

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
