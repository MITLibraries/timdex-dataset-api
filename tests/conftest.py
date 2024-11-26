import pytest

from timdex_dataset_api import TIMDEXDataset


@pytest.fixture(autouse=True)
def _test_env(monkeypatch):
    monkeypatch.setenv("TDA_LOG_LEVEL", "INFO")


@pytest.fixture
def local_dataset_location():
    return "tests/fixtures/local_datasets/dataset"


@pytest.fixture
def local_dataset(local_dataset_location):
    return TIMDEXDataset.load(local_dataset_location)
