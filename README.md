from timdex_dataset_api import TIMDEXDataset

# timdex-dataset-api
Python library for interacting with a TIMDEX parquet dataset located remotely or in S3.  This library is often abbreviated as "TDA".

## Development

- To preview a list of available Makefile commands: `make help`
- To install with dev dependencies: `make install`
- To update dependencies: `make update`
- To run unit tests: `make test`
- To lint the repo: `make lint`

The library version number is set in [`timdex_dataset_api/__init__.py`](timdex_dataset_api/__init__.py), e.g.:
```python
__version__ = "2.1.0"
```

Updating the version number when making changes to the library will prompt applications that install it, when they have _their_ dependencies updated, to pickup the new version.

## Installation

This library is designed to be utilized by other projects, and can therefore be added as a dependency directly from the Github repository.

Add via `pipenv`:
```shell
pipenv add git+https://github.com/MITLibraries/timdex-dataset-api.git
```

Manually add to `Pipfile`:
```
[packages]
... other dependencies...
timdex_dataset_api = {git = "https://github.com/MITLibraries/timdex-dataset-api.git"}
... other dependencies...
```

## Environment Variables

### Required

None at this time.

### Optional
```shell
TDA_LOG_LEVEL=# log level for timdex-dataset-api, accepts [DEBUG, INFO, WARNING, ERROR], default INFO
WARNING_ONLY_LOGGERS=# Comma-seperated list of logger names to set as WARNING only, e.g. 'botocore,charset_normalizer,smart_open'

MINIO_S3_ENDPOINT_URL=# If set, informs the library to use this Minio S3 instance.  Requires the http(s):// protocol. 
MINIO_USERNAME=# Username / AWS Key for Minio; required when MINIO_S3_ENDPOINT_URL is set
MINIO_PASSWORD=# Pasword / AWS Secret for Minio; required when MINIO_S3_ENDPOINT_URL is set
MINIO_DATA=# Path to persist MinIO data if started via Makefile command

TDA_READ_BATCH_SIZE=# Row size of batches read, affecting memory consumption
TDA_WRITE_BATCH_SIZE=# Row size of batches written, directly affecting row group size in final parquet files
TDA_MAX_ROWS_PER_GROUP=# Max number of rows per row group in a parquet file
TDA_MAX_ROWS_PER_FILE=# Max number of rows in a single parquet file
TDA_BATCH_READ_AHEAD=# Number of batches to optimistically read ahead when batch reading from a dataset; pyarrow default is 16
TDA_FRAGMENT_READ_AHEAD=# Number of fragments to optimistically read ahead when batch reaching from a dataset; pyarrow default is 4
TDA_DUCKDB_MEMORY_LIMIT=# Memory limit for DuckDB connection
TDA_DUCKDB_THREADS=# Thread limit for DuckDB connection
TDA_DUCKDB_JOIN_BATCH_SIZE=# Batch size for metadata + data joins, 100k default and recommended
```

## Local S3 via MinIO

Set env vars:
```shell
MINIO_S3_ENDPOINT_URL=http://localhost:9000
MINIO_USERNAME="admin"
MINIO_PASSWORD="password"
MINIO_DATA=<path to persist MinIO data, e.g. /tmp/minio>
```

Use a `Makefile` command that will start a MinIO instance:

```shell
make minio-start
```

With the env var `MINIO_S3_ENDPOINT_URL` set, this library will configure `pyarrow` and DuckDB connections to point at this local MinIO S3 instance.

## Usage

Currently, the most common use cases are:
  * **Transmogrifier**: uses TDA to **write** to the parquet dataset
  * **TIMDEX-Index-Manager (TIM)**: uses TDA to **read** from the parquet dataset

Beyond those two ETL run use cases, others are emerging where this library proves helpful:

  * yielding only the current version of all records in the dataset, useful for quickly re-indexing to Opensearch
  * high throughput (time) + memory safe (space) access to the dataset for analysis

For both reading and writing, the following env vars are recommended:
```shell
TDA_LOG_LEVEL=INFO
WARNING_ONLY_LOGGERS=asyncio,botocore,urllib3,s3transfer,boto3
```

### Reading Data

See [docs/reading.md](docs/reading.md) for an in-depth guide and Mermaid diagram.

First, import the library:
```python
from timdex_dataset_api import TIMDEXDataset
```

Load a dataset instance:
```python
# dataset in S3
timdex_dataset = TIMDEXDataset("s3://my-bucket/path/to/dataset")

# or, local dataset (e.g. testing or development)
timdex_dataset = TIMDEXDataset("/path/to/dataset")
```

All read methods for `TIMDEXDataset` allow for the same group of filters which are defined in `timdex_dataset_api.dataset.DatasetFilters`.  Examples are shown below.

```python
# read a single row, no filtering
single_record_dict = next(timdex_dataset.read_dicts_iter())


# get batches of records, filtering to a particular run
for batch in timdex_dataset.read_batches_iter(
    source="alma",
    run_date="2025-06-01",
    run_id="abc123"
):
    # do thing with pyarrow batch...


# use convenience method to yield only transformed records
# NOTE: this is what TIM uses for indexing to Opensearch for a given ETL run
for transformed_record in timdex_dataset.read_transformed_records_iter(
    source="aspace",
    run_date="2025-06-01",
    run_id="ghi789"
):
    # do something with transformed record dictionary...


# load all records for a given run into a pandas dataframe
# NOTE: this can be potentially expensive memory-wise if the run is large
run_df = timdex_dataset.read_dataframe(
    source="dspace",
    run_date="2025-06-01",
    run_id="def456"
)
```

### Writing Data

At this time, the only application that writes to the ETL parquet dataset is Transmogrifier.

To write records to the dataset, you must prepare an iterator of `timdex_dataset_api.record.DatasetRecord`.  Here is some pseudocode for how a dataset write can work:

```python
from timdex_dataset_api import DatasetRecord, TIMDEXDataset

# different ways to achieve, just need some kind of iterator (e.g. list, generator, etc.)
# of DatasetRecords for writing
def records_to_write_iter() -> Iterator[DatasetRecord]:	
    records = [...]		
	for record in records:
        yield DatasetRecord(
            timdex_record_id=...,
            source_record=...,
            transformed_record=...,
            source=...,
            run_date=...,
            run_type=...,
            run_timestamp=...,
            action=...,
            run_record_offset=...			
        )
records_iter = records_to_write_iter()
    
# finally, perform the write, relying on the library to handle efficient batching
timdex_dataset = TIMDEXDataset("/path/to/dataset")
timdex_dataset.write(records_iter=records_iter)
```