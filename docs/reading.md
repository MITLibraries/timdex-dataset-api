# Reading data from TIMDEXDataset

This guide explains how TIMDEXDataset data source read methods work and how to use them effectively.

- `TIMDEXDataset` maintains an in-memory DuckDB context. You can issue DuckDB SQL against the views/tables they create.
- Source-specific read methods are exposed on `timdex_dataset.records` and `timdex_dataset.embeddings`.
- Read methods use a two-step query flow for performance:
  1) a metadata query determines which Parquet files and row offsets are relevant
  2) a data query reads just those rows and returns the requested columns
- Prefer simple key/value filters for most use cases; add a `where=` SQL predicate when you need more advanced logic (e.g., ranges, `BETWEEN`, `>`, `<`, `IN`).

## Available read methods

The shared read methods below are available on both `timdex_dataset.records` and
`timdex_dataset.embeddings`:

- `read_batches_iter(...)`: yields `pyarrow.RecordBatch`
- `read_dicts_iter(...)`: yields Python `dict` per row
- `read_dataframe(...)`: returns a pandas `DataFrame`
- `read_dataframes_iter(...)`: yields pandas `DataFrame` batches

Additionally, `timdex_dataset.records` provides:

- `read_transformed_records_iter(...)`: yields `transformed_record` dictionaries only

All accept the same key/value filters and the optional `where=` SQL predicate.

## Filters vs. where=

- Key/value filters are keyword arguments on read methods. They are validated and translated into SQL and will cover most queries.
  - Examples: `source="alma"`, `run_date="2024-12-01"`, `run_type="daily"`, `action="index"`
- `where=` is an optional raw SQL WHERE predicate string, combined with these filters using `AND`. Use it for:
  - date/time ranges (BETWEEN, >, <)
  - set membership (IN (...))
  - complex boolean logic (AND/OR grouping)

Important: `where=` must be only a WHERE predicate (no `SELECT`/`FROM`/`;`). The library plugs it into generated SQL.

## How reading works (two-step process)

1) Metadata query
   - Runs against `TIMDEXDatasetMetadata` views (e.g., `metadata.records`, `metadata.current_records`)
   - Produces a small result set with identifiers: `filename`, row group/offsets, and primary keys
   - Greatly reduces how much data must be scanned

2) Data query
   - Uses DuckDB to read only relevant Parquet fragments based on metadata results
   - Joins the metadata identifiers to return the exact rows requested
   - Returns batches, dicts, or a `DataFrame` depending on the method

This pattern keeps reads fast and memory-efficient even for large datasets.

The following diagram shows the flow for an example query:

```python
for record_dict in td.records.read_dicts_iter(
    table="records",
    source="dspace",
    run_date="2025-09-01",
    run_id="abc123"
):
    # process record...
```

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant TD as TIMDEXDataset
    participant TDM as TIMDEXDatasetMetadata
    participant D as DuckDB Context
    participant P as Parquet files

    U->>TD: Perform query
    Note left of TD: records.read_dicts_iter(<br>table="records",<br>source="dspace",<br>run_date="2025-09-01",<br>run_id="abc123")
    TD->>TDM: build_meta_query(table, filters, where=None)
    Note right of TDM: (Metadata Query)<br><br>SELECT r.timdex_record_id, r.run_id, r.filename, r.run_record_offset<br>FROM metadata.records r<br>WHERE r.source = 'dspace'<br>AND r.run_date = '2025-09-01'<br>AND r.run_id = 'abc123'<br>ORDER BY r.filename, r.run_record_offset

    TDM->>D: Execute metadata query
    D-->>TD: lightweight result set (file + offsets)

    TD->>D: Build and run data query using metadata
    Note right of D: (Data query)<br><br>SELECT <COLUMNS><br>FROM read_parquet(P.files) d<br>JOIN meta m<br>USING (timdex_record_id, run_id, run_record_offset)

    D-->>TD: batches of rows
    TD-->>U: iterator of dicts (one dict per row)
```


## Quick start examples

```python
from timdex_dataset_api import TIMDEXDataset

td = TIMDEXDataset("s3://my-bucket/timdex-dataset")  # example instance

# 1) Get a single record as a dict
first = next(td.records.read_dicts_iter())

# 2) Read batches with simple filters
for batch in td.records.read_batches_iter(
    source="alma",
    run_date="2025-06-01",
    run_id="abc123",
):
    ...  # process pyarrow.RecordBatch

# 3) DataFrame of one run
df = td.records.read_dataframe(
    source="dspace",
    run_date="2025-06-01",
    run_id="def456",
)

# 4) Only transformed records (used by indexer)
for rec in td.records.read_transformed_records_iter(
    source="aspace",
    run_type="daily",
):
    ...  # rec is a dict of the transformed_record
```

## `where=` examples

Advanced filtering that complements key/value filters.

```python
# date range with BETWEEN
where = "run_date BETWEEN '2024-12-01' AND '2024-12-31'"
df = td.records.read_dataframe(source="alma", where=where)

# greater-than on a timestamp (if present in columns)
where = "run_timestamp > '2024-12-01T10:00:00Z'"
df = td.records.read_dataframe(source="aspace", run_type="daily", where=where)

# combine set membership and action
where = "run_id IN ('run-1', 'run-3', 'run-5') AND action = 'index'"
df = td.records.read_dataframe(source="alma", where=where)

# combine filters (AND) with where=
where = "run_type = 'daily' AND action = 'index'"
df = td.records.read_dataframe(source="libguides", where=where)
```

Validation tips:
- Use only a predicate (no SELECT/FROM, no trailing semicolon).
- Column names must exist in the target table/view (e.g., records or current_records).
- Key/value filters + `where=` are ANDed; if the combination yields zero rows, you’ll get an empty result.

## Choosing a table

For `timdex_dataset.records`, read methods query the `records` table by default (all versions). To get only the latest version per `timdex_record_id`, target the `current_records` view:

```python
# ALL records in the 'libguides' source
all_libguides_df = td.records.read_dataframe(table="records", source="libguides")

# latest unique records across the dataset
current_df = td.records.read_dataframe(table="current_records")

# current records for a source and specific run
current_df = td.records.read_dataframe(
    table="current_records",
    source="alma",
    run_id="run-5",
)
```

## DuckDB context

- `TIMDEXDataset` exposes a DuckDB connection used for metadata and data queries against Parquet.
- `TIMDEXDataSource` provides a base class that data sources extend
  - each data source class defines "tables" that are available for that source in the `metadata` schema

You can execute raw DuckDB SQL for inspection and debugging:

```python
# access dataset DuckDB connection
conn = td.conn  # DuckDB connection

# peek at view schemas
print(conn.sql("DESCRIBE metadata.records").to_df())
print(conn.sql("DESCRIBE metadata.current_records").to_df())

# ad-hoc query (read-only)
debug_df = conn.sql("""
    SELECT source, action, COUNT(*) as n
    FROM metadata.records
    WHERE run_date = '2024-12-01'
    GROUP BY 1, 2
    ORDER BY n DESC
""").to_df()
```

## Performance notes

- Batch iterators (`read_batches_iter()` / `read_dataframes_iter()`) stream results to control memory.
- `read_dataframe()` loads ALL matching rows into memory; fine for small/filtered sets but can easily overwhelm memory for large result sets
- Tuning via env vars (advanced): `TDA_READ_BATCH_SIZE`, `TDA_DUCKDB_THREADS`, `TDA_DUCKDB_MEMORY_LIMIT`.

## Troubleshooting

- Empty results? Check that filters and `where=` don’t over-constrain your query.
- Syntax errors? Ensure `where=` is a valid predicate and references existing columns.
- Large scans? Make sure to use `_iter()` read methods.
