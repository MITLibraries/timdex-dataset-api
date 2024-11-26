import pytest


@pytest.fixture(autouse=True)
def _test_env(monkeypatch):
    monkeypatch.setenv("TDA_LOG_LEVEL", "INFO")
