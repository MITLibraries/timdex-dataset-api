# ruff: noqa: BLE001, D212, TRY300, TRY400
"""
Date: 2025-06-25

Description:

This migration will ensure that all rows, for all parquet files, for a given ETL run have
the same run_timestamp.

When migration 001 was performed, it did not take into consideration that ETL runs with
100k+ records -- e.g. full Alma or DSpace -- would span multiple parquet files.  With
multiple parquet files came multiple S3 object dates which is what was used to backfill
that run_timestamp column.

This discovery led to TIMX-509, which ensures that a single run_timestamp can be applied
to all rows / files for a given ETL run.  Now that TIMX-509 is complete and deployed,
ensuring all *future* rows are written correctly, this migration is needed to update
a subset of run_timestamp values from migration 001.

The approach is fairly simple and uses the new TIMDEXDatasetMetadata class:

1. retrieve metadata for all records
2. for a given ETL run (run_id), find the earliest run_timestamp
3. apply that run_timestamp to all rows / files for that run_id

Usage:

PYTHONPATH=. \
pipenv run python migrations/002_2025_06_25_consistent_run_timestamp_per_etl_run.py \
<DATASET_LOCATION> \
--dry-run
"""

import argparse
import json
import time
from datetime import datetime

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from timdex_dataset_api import TIMDEXDatasetMetadata
from timdex_dataset_api.config import configure_dev_logger, configure_logger
from timdex_dataset_api.dataset import TIMDEX_DATASET_SCHEMA, TIMDEXDataset

configure_dev_logger()

logger = configure_logger(__name__)


def fix_backfilled_run_timestamps(location: str, *, dry_run: bool = False) -> None:
    """Main entrypoint for backfill script."""
    start_time = time.perf_counter()
    td = TIMDEXDataset(location)
    td.load()

    parquet_to_run_timestamp_df = prepare_run_timestamps_for_select_parquet_files(td)

    success_count = 0
    skip_count = 0
    error_count = 0

    for idx, row in parquet_to_run_timestamp_df.iterrows():

        if row.status == "OK":
            continue

        logger.info(
            f"Working on parquet file {int(idx) + 1}/{len(parquet_to_run_timestamp_df)}- "  # type: ignore[call-overload]
            f"run_id: {row.run_id}, status: {row.status}, filename: {row.filename}"
        )

        success, result = backfill_parquet_file(
            row.filename, td.dataset, row.earliest_timestamp, dry_run=dry_run
        )

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


def prepare_run_timestamps_for_select_parquet_files(
    timdex_dataset: TIMDEXDataset,
) -> pd.DataFrame:
    """Prepare a dataframe of parquet file to earliest + current timestamp for run_id.

    Returns:
        pd.DataFrame

    Example row:

        mapping.loc[".../df65fb2d-a071-4d96-87cf-c1288a5e010f-1.parquet"]

        ROW:
        run_id                2b9baa22-8eb8-41ff-8108-bd247ae884bd
        earliest_timestamp               2025-02-28 13:02:07-05:00
        current_timestamp                2025-02-28 13:33:46-05:00
        record_count                                         45025
        status                                              UPDATE
        Name:   .../df65fb2d-a071-4d96-87cf-c1288a5e010f-1.parquet

    """
    tdm = TIMDEXDatasetMetadata(timdex_dataset=timdex_dataset)

    query = """
    with earliest_timestamps as (
        select
            run_id,
            min(run_timestamp) as earliest_timestamp
        from records
        group by run_id
    )
    select
        r.run_id,
        et.earliest_timestamp,
        r.run_timestamp as current_timestamp,
        regexp_replace(r.filename, '^s3://', '') as filename,
        count(*) as record_count,
        case
            when r.run_timestamp = et.earliest_timestamp then 'ok'
            else 'update'
        end as status
    from records r
    join earliest_timestamps et on r.run_id = et.run_id
    group by r.run_id, et.earliest_timestamp, r.run_timestamp, r.filename
    order by r.run_timestamp, r.run_id;
    """
    return tdm.conn.query(query).to_df()


def backfill_parquet_file(
    parquet_filepath: str,
    dataset: ds.Dataset,
    new_run_timestamp: datetime,
    *,
    dry_run: bool = False,
) -> tuple[bool, dict]:
    """Backfill a single parquet file with the correct run_timestamp value for ETL run.

    Args:
        parquet_filepath: Path to the parquet file
        dataset: PyArrow dataset instance
        new_run_timestamp: datetime
        dry_run: If True, don't actually write changes

    Returns:
        Tuple of (success: bool, result: dict)
    """
    start_time = time.perf_counter()
    try:
        parquet_file = pq.ParquetFile(parquet_filepath, filesystem=dataset.filesystem)  # type: ignore[attr-defined]

        # read all rows from the parquet file into a pyarrow Table
        table = parquet_file.read()

        # set new run_timestamp value
        num_rows = len(table)
        run_timestamp_field = TIMDEX_DATASET_SCHEMA.field("run_timestamp")
        new_run_timestamp_array = pa.array(
            [new_run_timestamp] * num_rows, type=run_timestamp_field.type
        )
        table_updated = table.set_column(
            table.schema.get_field_index("run_timestamp"),
            "run_timestamp",
            new_run_timestamp_array,
        )

        # write the updated table back to the same file
        if not dry_run:
            pq.write_table(
                table_updated,  # type: ignore[attr-defined]
                parquet_filepath,
                filesystem=dataset.filesystem,  # type: ignore[attr-defined]
            )
            logger.info(f"Successfully updated file: {parquet_filepath}")
        else:
            logger.info(f"DRY RUN: Would update file: {parquet_filepath}")

        update_details = {
            "file_path": parquet_filepath,
            "rows_updated": num_rows,
            "new_run_timestamp": new_run_timestamp.isoformat(),
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Ensures each ETL run has a single run_timestamp value.")
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan files and report what would be done without making changes",
    )
    parser.add_argument(
        "dataset_location",
        help="Path to the dataset (local path or s3://bucket/path)",
    )

    args = parser.parse_args()

    fix_backfilled_run_timestamps(args.dataset_location, dry_run=args.dry_run)
