"""timdex_dataset_api/exceptions.py"""


class DatasetNotLoadedError(Exception):
    """Custom exception for accessing methods requiring a loaded dataset."""


class InvalidDatasetRecordError(Exception):
    """Custom exception for invalid DatasetRecord instances."""
