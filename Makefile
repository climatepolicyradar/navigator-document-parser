.PHONY: run_local run_docker install test_local build test

install:
	poetry install
	poetry run pre-commit install
	poetry run playwright install 
	poetry run pip install "git+https://github.com/facebookresearch/detectron2.git@v0.5#egg=detectron2"
	cp .env.example .env

run_local:
	LAYOUTPARSER_MODEL=faster_rcnn_R_50_FPN_3x PDF_OCR_AGENT=tesseract TARGET_LANGUAGES=en GOOGLE_APPLICATION_CREDENTIALS=./credentials/google-creds.json python -m cli.run_parser ./data/raw ./data/processed

test_local:
	LAYOUTPARSER_MODEL=faster_rcnn_R_50_FPN_3x PDF_OCR_AGENT=tesseract TARGET_LANGUAGES=en CDN_DOMAIN=cdn.climatepolicyradar.org python -m pytest -vvv

build:
	docker build -t navigator-document-parser .

test:
	docker build -t navigator-document-parser .
	docker run -e "LAYOUTPARSER_MODEL=faster_rcnn_R_50_FPN_3x" -e "PDF_OCR_AGENT=tesseract" -e "CDN_DOMAIN=cdn.climatepolicyradar.org" --network host navigator-document-parser python -m pytest -vvv

run_docker:
	docker build -t html-parser .
	docker run --network host -v ${PWD}/data:/app/data html-parser python -m cli.run_parser ./data/raw ./data/processed

run_on_specific_files_flag:
	docker build -t html-parser .
	docker run -it html-parser python -m cli.run_parser s3://cpr-dev-data-pipeline-cache/marks/ingest_output/ s3://cpr-dev-data-pipeline-cache/marks/parser_output/ --s3 --files "1331.0.json"

run_on_specific_files_env:
	docker build -t html-parser .
	docker run -it -e files_to_parse="$1331.0.json" html-parser python -m cli.run_parser s3://cpr-dev-data-pipeline-cache/marks/ingest_output/ s3://cpr-dev-data-pipeline-cache/marks/parser_output/ --s3

run_local_against_s3:
	docker build -t html-parser .
	docker run -e PARSER_INPUT_PREFIX=s3://cpr-dev-data-pipeline-cache/marks/ingest_output/ -e EMBEDDINGS_INPUT_PREFIX=s3://cpr-dev-data-pipeline-cache/marks/parser_output/ -it html-parser

build_and_push_ecr:
	aws ecr get-login-password --region eu-west-2 --profile dev | docker login --username AWS --password-stdin 073457443605.dkr.ecr.eu-west-2.amazonaws.com
	docker build -t navigator-document-parser-staging .
	docker tag navigator-document-parser-staging:latest 073457443605.dkr.ecr.eu-west-2.amazonaws.com/navigator-document-parser-staging:latest
	docker push 073457443605.dkr.ecr.eu-west-2.amazonaws.com/navigator-document-parser-staging:latest
