"""tests/test_s3client.py"""

# ruff: noqa: PLR2004, SLF001


import boto3
import moto
import pytest

from timdex_dataset_api.utils import S3Client


@pytest.fixture
def mock_s3_resource():
    """Set up a mocked S3 resource using moto."""
    with moto.mock_aws():
        # Create a test bucket
        conn = boto3.resource("s3", region_name="us-east-1")
        conn.create_bucket(Bucket="test-bucket")
        yield conn


@pytest.fixture
def s3_client():
    """Return an S3Client instance."""
    return S3Client()


def test_s3client_init():
    """Test S3Client initialization."""
    client = S3Client()
    assert client.resource is not None


def test_s3client_init_with_minio_env(monkeypatch):
    """Test S3Client initialization with MinIO environment variables."""
    monkeypatch.setenv("MINIO_S3_ENDPOINT_URL", "http://localhost:9000")
    monkeypatch.setenv("MINIO_USERNAME", "minioadmin")
    monkeypatch.setenv("MINIO_PASSWORD", "minioadmin")
    monkeypatch.setenv("MINIO_REGION", "us-east-1")

    client = S3Client()
    assert client.resource is not None


def test_split_s3_uri():
    """Test _split_s3_uri method."""
    client = S3Client()
    bucket, key = client._split_s3_uri("s3://test-bucket/path/to/file.txt")
    assert bucket == "test-bucket"
    assert key == "path/to/file.txt"


def test_split_s3_uri_invalid():
    """Test _split_s3_uri method with invalid URI."""
    client = S3Client()
    with pytest.raises(ValueError, match="Invalid S3 URI"):
        client._split_s3_uri("test-bucket/path/to/file.txt")


def test_upload_download_file(mock_s3_resource, tmp_path):
    """Test upload_file and download_file methods."""
    client = S3Client()

    # Create a test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")

    # Upload the file
    s3_uri = "s3://test-bucket/test.txt"
    client.upload_file(test_file, s3_uri)

    # Download the file to a different location
    download_path = tmp_path / "downloaded.txt"
    client.download_file(s3_uri, download_path)

    # Verify the content
    assert download_path.read_text() == "test content"


def test_delete_file(mock_s3_resource, tmp_path):
    """Test delete_file method."""
    client = S3Client()

    # Create and upload a test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")
    s3_uri = "s3://test-bucket/test.txt"
    client.upload_file(test_file, s3_uri)

    # Delete the file
    client.delete_file(s3_uri)

    # Verify the file is deleted
    bucket = mock_s3_resource.Bucket("test-bucket")
    objects = list(bucket.objects.all())
    assert len(objects) == 0


def test_delete_folder(mock_s3_resource, tmp_path):
    """Test delete_folder method."""
    client = S3Client()

    # Create and upload test files
    for i in range(3):
        test_file = tmp_path / f"test{i}.txt"
        test_file.write_text(f"test content {i}")
        s3_uri = f"s3://test-bucket/folder/test{i}.txt"
        client.upload_file(test_file, s3_uri)

    # Upload a file outside the folder
    other_file = tmp_path / "other.txt"
    other_file.write_text("other content")
    client.upload_file(other_file, "s3://test-bucket/other.txt")

    # Delete the folder
    deleted_keys = client.delete_folder("s3://test-bucket/folder/")

    # Verify only folder contents are deleted
    assert len(deleted_keys) == 3
    assert all(key.startswith("folder/") for key in deleted_keys)

    bucket = mock_s3_resource.Bucket("test-bucket")
    objects = list(bucket.objects.all())
    assert len(objects) == 1
    assert objects[0].key == "other.txt"
