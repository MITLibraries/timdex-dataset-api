# TIMDEX Dataset Migrations

This directory includes manual, bulk migrations of data and schema in the TIMDEX parquet dataset.  Consider it like migrations for a SQL database, except a bit more unstructured and ad-hoc.

##  Structure

Each migration is either a single python file, or a dedicated directory, with that follows the naming convention:

  - `###_`: incrementing migration sequence number
  - `YYYY_MM_DD_`: approximate date of migration creation and run
  - `short_name.py` (file) or `short_name` (directory): short migration name

Examples:

  - `001_2025_05_30_backfill_run_timestamp_column.py` --> single file
  - `002_2025_06_15_remove_errant_parquet_files` --> directory that contains 1+ files

The entrypoint for each migration should contain a docstring at the root of the file with a structure like:

```python
"""
Date: YYYY-MM-DD

Description:

Description here about the nature of the migration...

Usage:

Explanation here for how to run it...
"""
```

Example:
```python
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
PYTHONPATH=. \
pipenv run python migrations/001_2025_05_30_backfill_run_timestamp_column.py \
<DATASET_LOCATION> \
--dry-run
"""
```