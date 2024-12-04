# timdex-dataset-api
Python library for interacting with a TIMDEX parquet dataset located remotely or in S3.

## Development

- To preview a list of available Makefile commands: `make help`
- To install with dev dependencies: `make install`
- To update dependencies: `make update`
- To run unit tests: `make test`
- To lint the repo: `make lint`

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

### Optional
```shell
TDA_LOG_LEVEL=# log level for timdex-dataset-api, accepts [DEBUG, INFO, WARNING, ERROR], default INFO
```

## Usage

_TODO..._