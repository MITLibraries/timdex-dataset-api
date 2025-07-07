# ruff: noqa: BLE001, D212, RUF002, RUF001, TRY400

"""
Date: 2025-07-03

Description:

Re‑pack any Parquet file in a TIMDEX dataset that still has **one** oversized row‑group
so that no group exceeds the configured `TDA_MAX_ROWS_PER_GROUP` (defaults to 1k).

Usage
-----
PYTHONPATH=. \
pipenv run python migrations/003_2025_07_03_fix_row_group_sizes.py \
<DATASET_LOCATION> \
--dry-run
"""

import argparse
import logging
import time
from math import ceil

import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pyarrow import fs

from timdex_dataset_api.config import configure_dev_logger
from timdex_dataset_api.dataset import (
    TIMDEXDataset,
    TIMDEXDatasetConfig,
)

configure_dev_logger()

DEFAULT_ROWS_PER_GROUP: int = TIMDEXDatasetConfig().max_rows_per_group
DEFAULT_COMPRESSION: str | None = "snappy"

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def needs_rewrite(pf: pq.ParquetFile, rows_per_group: int) -> bool:
    return pf.metadata.num_row_groups == 1 and pf.metadata.num_rows > rows_per_group


def rewrite_file(
    path: str,
    filesystem: fs.FileSystem,
    *,
    dry_run: bool = False,
) -> None:
    pf = pq.ParquetFile(path, filesystem=filesystem)

    codec = pf.metadata.row_group(0).column(0).compression
    compression = None if codec.upper() == "UNCOMPRESSED" else codec.lower()

    if dry_run:
        log.info(
            f"[DRY‑RUN] {path} — would split {pf.metadata.num_rows:,} rows "
            f"into {ceil(pf.metadata.num_rows / DEFAULT_ROWS_PER_GROUP):,} row‑groups"
        )
        return

    table = pf.read()

    pq.write_table(
        table,
        path,
        filesystem=filesystem,
        row_group_size=DEFAULT_ROWS_PER_GROUP,
        compression=compression,  # type: ignore[arg-type]
    )

    log.info(
        f"[OK] {path} — {pf.metadata.num_rows:,} rows now split into "
        f"{ceil(pf.metadata.num_rows / DEFAULT_ROWS_PER_GROUP):,} row‑groups"
    )


def fix_row_groups(
    location: str,
    *,
    dry_run: bool = False,
) -> None:
    start = time.perf_counter()

    filesystem, dataset_root = TIMDEXDataset.parse_location(location)
    if isinstance(dataset_root, list):
        raise TypeError("Expected a single dataset root path, not a list.")

    dataset = ds.dataset(dataset_root, format="parquet", filesystem=filesystem)

    rewritten = skipped = errors = 0
    for file_path in dataset.files:
        try:
            pf = pq.ParquetFile(file_path, filesystem=filesystem)
            if needs_rewrite(pf, DEFAULT_ROWS_PER_GROUP):
                rewrite_file(
                    file_path,
                    filesystem,
                    dry_run=dry_run,
                )
                rewritten += 1
            else:
                skipped += 1
        except Exception as exc:
            log.error(f"[ERR] {file_path}: {exc}")
            errors += 1

    elapsed = time.perf_counter() - start
    log.info(
        f"Done in {elapsed:.1f}s — rewritten: {rewritten}, "
        f"skipped: {skipped}, errors: {errors}"
    )


def _parse_args() -> tuple[str, bool]:
    p = argparse.ArgumentParser(
        description="Re‑pack Parquet files so no row‑group exceeds the configured size."
    )
    p.add_argument("dataset_location")
    p.add_argument("--dry-run", action="store_true")

    a = p.parse_args()
    return a.dataset_location, a.dry_run


if __name__ == "__main__":
    location, dry_run = _parse_args()
    fix_row_groups(location, dry_run=dry_run)
