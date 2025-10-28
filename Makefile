.PHONY: run_local run_docker install test_local build test
-include .env

setup:
	cp .env.example .env

install:
	poetry env activate
	poetry install

run_local:
	TARGET_LANGUAGES=en GOOGLE_APPLICATION_CREDENTIALS=./credentials/google-creds.json python -m cli.run_parser ./data/raw ./data/processed --document_import_ids "${DOCUMENT_IMPORT_IDS}"

test_local:
	TARGET_LANGUAGES=en poetry run python -m pytest -vvv

build:
	docker build -t navigator-document-parser .
	docker tag navigator-document-parser navigator-document-parser-staging

pre-commit-checks-all-files:
	docker run --entrypoint pre-commit navigator-document-parser run --all-files

test:
	docker build -t navigator-document-parser .
	docker run -e AZURE_PROCESSOR_KEY="${AZURE_PROCESSOR_KEY}" -e AZURE_PROCESSOR_ENDPOINT="${AZURE_PROCESSOR_ENDPOINT}" --network host --entrypoint python3 navigator-document-parser -m pytest -vvv

run_docker:
	docker build -t navigator-document-parser .
	docker run --network host -v ${PWD}/data:/app/data -e AZURE_PROCESSOR_KEY="${AZURE_PROCESSOR_KEY}" -e AZURE_PROCESSOR_ENDPOINT="${AZURE_PROCESSOR_ENDPOINT}" -e GOOGLE_CREDS="${GOOGLE_CREDS}" navigator-document-parser ./data/raw ./data/processed "${DOCUMENT_IMPORT_IDS}"

run_local_against_s3:
	docker build -t navigator-document-parser .
	docker run -e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" -e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" -e AZURE_PROCESSOR_KEY="${AZURE_PROCESSOR_KEY}" -e AZURE_PROCESSOR_ENDPOINT="${AZURE_PROCESSOR_ENDPOINT}" -e GOOGLE_CREDS="${GOOGLE_CREDS}" -it navigator-document-parser "${PARSER_INPUT_PREFIX}" "${PARSER_OUTPUT_PREFIX}" "${DOCUMENT_IMPORT_IDS}"
