-include .env

SHELL=/bin/bash
DATETIME:=$(shell date -u +%Y%m%dT%H%M%SZ)

help: # Preview Makefile commands
	@awk 'BEGIN { FS = ":.*#"; print "Usage:  make <target>\n\nTargets:" } \
/^[-_[:alpha:]]+:.?*#/ { printf "  %-15s%s\n", $$1, $$2 }' $(MAKEFILE_LIST)

#######################
# Dependency commands
#######################

install: # Install Python dependencies
	pipenv install --dev
	pre-commit install --hook-type pre-push --hook-type pre-commit

update: install # Update Python dependencies
	pipenv clean
	pipenv update --dev

######################
# Unit test commands
######################

test: # Run tests and print a coverage report
	pipenv run coverage run --source=timdex_dataset_api -m pytest -vv
	pipenv run coverage report -m

coveralls: test # Write coverage data to an LCOV report
	pipenv run coverage lcov -o ./coverage/lcov.info

####################################
# Code linting and formatting
####################################

lint:
	pipenv run ruff format --diff
	pipenv run mypy .
	pipenv run ruff check .

lint-fix:
	pipenv run ruff format .
	pipenv run ruff check --fix .

security:
	pipenv run pip-audit


######################
# Minio S3 Instance
######################
minio-start:
	docker run \
	-d \
	-p 9000:9000 \
	-p 9001:9001 \
	-v $(MINIO_DATA):/data \
	-e "MINIO_ROOT_USER=$(MINIO_USERNAME)" \
	-e "MINIO_ROOT_PASSWORD=$(MINIO_PASSWORD)" \
	quay.io/minio/minio server /data --console-address ":9001"