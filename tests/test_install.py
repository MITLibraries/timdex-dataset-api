import timdex_dataset_api


def test_version_is_root_level_attribute():
    assert timdex_dataset_api.__version__ is not None
