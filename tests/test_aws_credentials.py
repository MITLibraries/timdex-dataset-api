"""tests/test_aws_credentials.py"""

# ruff: noqa: SLF001

import duckdb
import pytest
from botocore.stub import Stubber

from timdex_dataset_api.exceptions import AWSCredentialsError
from timdex_dataset_api.utils import DuckDBConnectionFactory, S3Client


@pytest.fixture
def no_aws_credentials(monkeypatch, tmp_path):
    """Isolate a test from all ambient AWS credential sources."""
    for env_var in [
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SECURITY_TOKEN",
    ]:
        monkeypatch.delenv(env_var, raising=False)

    aws_config_file = tmp_path / "config"
    aws_credentials_file = tmp_path / "credentials"
    aws_config_file.write_text("")
    aws_credentials_file.write_text("")

    monkeypatch.setenv("AWS_CONFIG_FILE", str(aws_config_file))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(aws_credentials_file))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")


def test_s3client_object_exists_access_denied_raises_aws_credentials_error(monkeypatch):
    """Test that 403 authorization errors bubble up as helpful TDA exceptions.

    This covers the scenario where AWS credentials ARE present, but point at the
    wrong account/environment or otherwise lack permission to read the dataset.
    Stubber exercises boto3/botocore's normal ClientError path without making a
    real AWS request.
    """
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "bad_access_key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "bad_secret_key")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "bad_session_token")

    client = S3Client()

    stubber = Stubber(client.resource.meta.client)
    stubber.add_client_error(
        "head_object",
        service_error_code="403",
        service_message="Forbidden",
        http_status_code=403,  # <------ simulates 403 authorization error
        expected_params={
            "Bucket": "timdex",
            "Key": "metadata/timdex_metadata.duckdb",
        },
    )

    with stubber, pytest.raises(AWSCredentialsError, match="AWS credentials absent"):
        client.object_exists("s3://timdex/metadata/timdex_metadata.duckdb")


def test_duckdb_s3_secret_without_credentials_raises_aws_credentials_error(
    no_aws_credentials,
):
    """Test that we get a helpful mesesage when DuckDB can't find AWS credentials.

    This test uses the local fixture 'no_aws_credentials' which completely unsets all
    traces of AWS credentials in the environment.  This is what expired credentials also
    look like to DuckDB when consructing a secret.
    """
    factory = DuckDBConnectionFactory(location_scheme="s3")

    with (
        duckdb.connect(":memory:") as conn,
        pytest.raises(
            AWSCredentialsError,
            match="AWS credentials absent",
        ),
    ):
        factory._configure_s3_secret(conn)
