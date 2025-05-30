"""timdex_dataset_api/run.py"""

import concurrent.futures
import logging
import time

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class TIMDEXRunManager:
    """Manages and provides access to ETL run metadata from the TIMDEX parquet dataset."""

    def __init__(self, dataset: ds.Dataset):
        self.dataset = dataset
        self._runs_metadata_cache: pd.DataFrame | None = None

    def clear_cache(self) -> None:
        self._runs_metadata_cache = None

    def get_runs_metadata(self, *, refresh: bool = False) -> pd.DataFrame:
        """Get metadata for all runs in dataset, grouped by run_id.

        The dataframe returned includes the following columns:
            - source
            - run_date
            - run_type
            - run_id
            - num_rows: total number of records for that run_id
            - parquet_files: list of parquet file(s) that are associated with that run

        Args:
            refresh: If True, force refresh of cached metadata
        """
        start_time = time.perf_counter()

        if self._runs_metadata_cache is not None and not refresh:
            return self._runs_metadata_cache

        ungrouped_runs_df = self._get_parquet_files_run_metadata()
        if ungrouped_runs_df.empty:
            return ungrouped_runs_df

        # group by run_id
        grouped_runs_df = (
            ungrouped_runs_df.groupby("run_id")
            .agg(
                {
                    "source": "first",
                    "run_date": "first",
                    "run_type": "first",
                    "run_timestamp": "first",
                    "num_rows": "sum",
                    "filename": list,
                }
            )
            .reset_index()
        )

        # add additional metadata
        grouped_runs_df = grouped_runs_df.rename(columns={"filename": "parquet_files"})
        grouped_runs_df["parquet_files_count"] = grouped_runs_df["parquet_files"].apply(
            lambda x: len(x)
        )

        # sort by run_timestamp (more granularity than run_date) and source
        grouped_runs_df = grouped_runs_df.sort_values(
            ["run_timestamp", "source"], ascending=False
        )

        # cache the result
        self._runs_metadata_cache = grouped_runs_df

        logger.info(
            f"Dataset runs metadata retrieved, elapsed: "
            f"{round(time.perf_counter() - start_time, 2)}s, runs: {len(grouped_runs_df)}"
        )
        return grouped_runs_df

    def get_current_parquet_files(self, source: str | None = None) -> list[str]:
        """Get reverse chronological list of parquet files associated with current runs.

        Args:
            source: if provided, limits parquet files to only that source
        """
        runs_df = self.get_runs_metadata()  # run metadata is cached for future calls
        sources = [source] if source else list(runs_df.source.unique())

        source_parquet_files = []
        for _source in sources:
            source_parquet_files.extend(self._get_current_source_parquet_files(_source))

        return source_parquet_files

    def _get_current_source_parquet_files(self, source: str) -> list[str]:
        """Get reverse chronological list of current parquet files for a source.

        Args:
            source: The source identifier to filter runs
        """
        runs_df = self.get_runs_metadata()
        source_runs_df = runs_df[runs_df.source == source].copy()

        # get last "full" run
        full_runs_df = source_runs_df[source_runs_df.run_type == "full"]
        if len(full_runs_df) == 0:
            raise RuntimeError(
                f"Could not find the most recent 'full' run for source: '{source}'"
            )
        last_full_run = full_runs_df.iloc[0]

        # get all "daily" runs since
        daily_runs_df = source_runs_df[
            (source_runs_df.run_type == "daily")
            & (source_runs_df.run_date >= last_full_run.run_date)
        ]

        ordered_parquet_files = []
        for _, daily_run in daily_runs_df.iterrows():
            ordered_parquet_files.extend(daily_run.parquet_files)
        ordered_parquet_files.extend(last_full_run.parquet_files)

        return ordered_parquet_files

    def _get_parquet_files_run_metadata(self, max_workers: int = 250) -> pd.DataFrame:
        """Retrieve run metadata from parquet file(s) in dataset.

        A single ETL run may still be spread across multiple Parquet files making this
        data ungrouped by run.

        Args:
            max_workers: Maximum number of parallel workers for processing
                - a high number is generally safe given the lightweight nature of the
                thread's work, just reading a few parquet file header bytes
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for parquet_filepath in self.dataset.files:  # type: ignore[attr-defined]
                future = executor.submit(
                    self._parse_run_metadata_from_parquet_file,
                    parquet_filepath,
                )
                futures.append(future)

            done, not_done = concurrent.futures.wait(
                futures, return_when=concurrent.futures.ALL_COMPLETED
            )

            results = []
            for future in done:
                try:
                    if result := future.result():
                        results.append(result)
                except Exception:
                    logger.exception("Error reading run metadata from parquet file.")

        return pd.DataFrame(results) if results else pd.DataFrame()

    def _parse_run_metadata_from_parquet_file(self, parquet_filepath: str) -> dict:
        """Parse source, run_date, run_type, and run_id from a single Parquet file.

        The TIMDEX parquet dataset has a characteristic that we can use for extracting
        run information from a single row in a parquet file: all rows in the parquet file
        share the column values source, run_date, run_type, and run_id.

        Taking this a step further, we can extract these values without even touching a
        single proper row from the parquet file, but from reading the parquet file
        column statistics.  In this way, we can extract run information from a parquet
        file by only reading the lightweight parquet file metadata.

        Args:
            parquet_filepath: Path to the parquet file
        """
        parquet_file = pq.ParquetFile(
            parquet_filepath,
            filesystem=self.dataset.filesystem,  # type: ignore[attr-defined]
        )

        file_meta = parquet_file.metadata.to_dict()
        num_rows = file_meta["num_rows"]
        columns_meta = file_meta["row_groups"][0]["columns"]  # type: ignore[typeddict-item]
        source = columns_meta[3]["statistics"]["max"]
        run_date = columns_meta[4]["statistics"]["max"]
        run_type = columns_meta[5]["statistics"]["max"]
        run_id = columns_meta[7]["statistics"]["max"]
        run_timestamp = columns_meta[9]["statistics"]["max"]

        return {
            "source": source,
            "run_date": run_date,
            "run_type": run_type,
            "run_id": run_id,
            "run_timestamp": run_timestamp,
            "num_rows": num_rows,
            "filename": parquet_filepath,
        }
