# ruff: noqa: BLE001, D212, TRY300, TRY400
"""
Date: 2025-05-30

Description:

After the creation of a new run_timestamp column as part of Jira ticket TIMX-496, there
was a need to backfill a run timestamp for all parquet files in the dataset.

This migration performs the following:
1. retrieves all parquet file from the dataset
2. for each parquet file:
    a. if the run_timestamp column already exists, skip
    b. retrieve the file creation date of the parquet file, this becomes the run_timestamp
    c. rewrite the parquet file with a new run_timestamp column

Usage:

pipenv run python migrations/001_2025_05_30_backfill_run_timestamp_column.py \
<DATASET_LOCATION> \
--dry-run
"""

import argparse
import json
import time
from datetime import UTC, datetime

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pyarrow import fs

from timdex_dataset_api.config import configure_dev_logger, configure_logger
from timdex_dataset_api.dataset import TIMDEX_DATASET_SCHEMA, TIMDEXDataset

configure_dev_logger()

logger = configure_logger(__name__)


def backfill_dataset(location: str, *, dry_run: bool = False) -> None:
    """Main entrypoint for backfill script.

    Loop through all parquet files in the dataset and, if the run_timestamp column does
    not exist, create it using the S3 object creation date.
    """
    start_time = time.perf_counter()
    td = TIMDEXDataset(location)
    td.load()

    parquet_files = td.dataset.files  # type: ignore[attr-defined]
    logger.info(f"Found {len(parquet_files)} parquet files in dataset.")

    success_count = 0
    skip_count = 0
    error_count = 0

    for i, parquet_file in enumerate(parquet_files):
        logger.info(
            f"Working on parquet file {i + 1}/{len(parquet_files)}: {parquet_file}"
        )

        success, result = backfill_parquet_file(parquet_file, td.dataset, dry_run=dry_run)

        if success:
            if result and "skipped" in result:
                skip_count += 1
            else:
                success_count += 1
        else:
            error_count += 1

        logger.info(json.dumps(result))

    logger.info(
        f"Backfill complete. Elapsed: {time.perf_counter()-start_time}, "
        f"Success: {success_count}, Skipped: {skip_count}, Errors: {error_count}"
    )


def backfill_parquet_file(
    parquet_filepath: str,
    dataset: ds.Dataset,
    *,
    dry_run: bool = False,
) -> tuple[bool, dict]:
    """Backfill a single parquet file with run_timestamp column.

    Args:
        parquet_filepath: Path to the parquet file
        dataset: PyArrow dataset instance
        dry_run: If True, don't actually write changes

    Returns:
        Tuple of (success: bool, result: dict)
    """
    start_time = time.perf_counter()
    try:
        parquet_file = pq.ParquetFile(parquet_filepath, filesystem=dataset.filesystem)  # type: ignore[attr-defined]

        # Check if run_timestamp column already exists
        if "run_timestamp" in parquet_file.schema.names:
            logger.info(
                f"Parquet already has 'run_timestamp', skipping: {parquet_filepath}"
            )
            return True, {"file_path": parquet_filepath, "skipped": True}

        # Read all rows from the parquet file into a pyarrow Table
        # NOTE: memory intensive for very large parquet files, though suitable for onetime
        #  migration work.
        table = parquet_file.read()

        # Get S3 object creation date
        creation_date = get_s3_object_creation_date(parquet_filepath, dataset.filesystem)  # type: ignore[attr-defined]

        # Create run_timestamp column using the exact schema definition
        num_rows = len(table)
        run_timestamp_field = TIMDEX_DATASET_SCHEMA.field("run_timestamp")
        run_timestamp_array = pa.array(
            [creation_date] * num_rows, type=run_timestamp_field.type
        )

        # Add the run_timestamp column to the table
        table_with_timestamp = table.append_column("run_timestamp", run_timestamp_array)

        # Write the updated table back to the same file
        if not dry_run:
            pq.write_table(
                table_with_timestamp,  # type: ignore[attr-defined]
                parquet_filepath,
                filesystem=dataset.filesystem,  # type: ignore[attr-defined]
            )
            logger.info(f"Successfully updated file: {parquet_filepath}")
        else:
            logger.info(f"DRY RUN: Would update file: {parquet_filepath}")

        update_details = {
            "file_path": parquet_filepath,
            "rows_updated": num_rows,
            "run_timestamp_added": creation_date.isoformat(),
            "elapsed": time.perf_counter() - start_time,
            "dry_run": dry_run,
        }

        return True, update_details

    except Exception as e:
        logger.error(f"Error processing parquet file {parquet_filepath}: {e}")
        return False, {
            "file_path": parquet_filepath,
            "error": str(e),
            "elapsed": time.perf_counter() - start_time,
            "dry_run": dry_run,
        }


def get_s3_object_creation_date(file_path: str, filesystem: fs.FileSystem) -> datetime:
    """Get the creation date of an S3 object.

    Args:
        file_path: Path to the S3 object
        filesystem: PyArrow S3 filesystem instance

    Returns:
        datetime: Creation date of the S3 object in UTC
    """
    try:
        # Get creation date of S3 object
        file_info = filesystem.get_file_info(file_path)
        creation_date: datetime = file_info.mtime  # type: ignore[assignment]

        # Ensure it's timezone-aware and in UTC
        if creation_date.tzinfo is None:
            creation_date = creation_date.replace(tzinfo=UTC)
        elif creation_date.tzinfo != UTC:
            creation_date = creation_date.astimezone(UTC)

        return creation_date

    except Exception as e:
        logger.error(f"Error getting S3 object creation date for {file_path}: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Backfill run_timestamp column in TIMDEX parquet files "
            "using S3 creation dates"
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan files and report what would be done without making changes",
    )
    parser.add_argument(
        "dataset_location", help="Path to the dataset (local path or s3://bucket/path)"
    )

    args = parser.parse_args()

    backfill_dataset(args.dataset_location, dry_run=args.dry_run)
