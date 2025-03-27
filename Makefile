.PHONY: run_local run_docker install test_local build test
-include .env

setup:
	cp .env.example .env

install:
	poetry shell
	poetry install

run_local:
	TARGET_LANGUAGES=en CDN_DOMAIN=cdn.climatepolicyradar.org GOOGLE_APPLICATION_CREDENTIALS=./credentials/google-creds.json python -m cli.run_parser ./data/raw ./data/processed

test_local:
	TARGET_LANGUAGES=en CDN_DOMAIN=cdn.climatepolicyradar.org poetry run python -m pytest -vvv

build:
	docker build -t navigator-document-parser .
	docker tag navigator-document-parser navigator-document-parser-staging

pre-commit-checks-all-files:
	docker run navigator-document-parser pre-commit run --all-files

test:
	docker build -t navigator-document-parser .
	docker run -e AZURE_PROCESSOR_KEY="${AZURE_PROCESSOR_KEY}" -e AZURE_PROCESSOR_ENDPOINT="${AZURE_PROCESSOR_ENDPOINT}" -e "CDN_DOMAIN=cdn.dev.climatepolicyradar.org" --network host navigator-document-parser python -m pytest -vvv

run_docker:
	docker build -t navigator-document-parser .
	docker run --network host -v ${PWD}/data:/app/data navigator-document-parser python -m cli.run_parser ./data/raw ./data/processed

run_on_specific_files_flag:
	docker build -t html-parser .
	docker run -it -e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" -e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" navigator-document-parser python -m cli.run_parser "${PARSER_INPUT_PREFIX}" "${EMBEDDINGS_INPUT_PREFIX}" --s3 --files "${FILES_TO_PARSE_FLAG}"

run_on_specific_files_env:
	docker build -t html-parser .
	docker run -it -e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" -e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" -e files_to_parse="${FILES_TO_PARSE}" navigator-document-parser python -m cli.run_parser "${PARSER_INPUT_PREFIX}" "${EMBEDDINGS_INPUT_PREFIX}" --s3

run_local_against_s3:
	docker build -t navigator-document-parser .
	docker run -e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" -e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" -e CDN_DOMAIN="${CDN_DOMAIN}" -e GOOGLE_CREDS="${GOOGLE_CREDS}" -e PARSER_INPUT_PREFIX="${PARSER_INPUT_PREFIX}" -e EMBEDDINGS_INPUT_PREFIX="${EMBEDDINGS_INPUT_PREFIX}" -it navigator-document-parser
