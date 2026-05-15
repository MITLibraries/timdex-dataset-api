"""timdex_dataset_api/exceptions.py"""


class AWSCredentialsError(Exception):
    """Raised when AWS credentials are absent or insufficient."""

    def __init__(
        self,
        message: str = (
            "AWS credentials absent or insufficient. Please ensure they are set "
            "and pointing at the right TIMDEX dataset environment."
        ),
    ) -> None:
        super().__init__(message)


class InvalidDatasetRecordError(Exception):
    """Custom exception for invalid DatasetRecord instances."""
