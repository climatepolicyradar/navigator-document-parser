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
	LAYOUTPARSER_MODEL=faster_rcnn_R_50_FPN_3x PDF_OCR_AGENT=tesseract TARGET_LANGUAGES=en python -m pytest

build:
	docker build -t html-parser .

test:
	docker run -e "LAYOUTPARSER_MODEL=faster_rcnn_R_50_FPN_3x" -e "PDF_OCR_AGENT=tesseract" --network host html-parser python -m pytest -vvv

run_docker:
	docker run --network host -v ${PWD}/data:/app/data html-parser python -m cli.run_parser ./data/raw ./data/processed

run_local_dotenv:
	docker build -t html-parser_local .
	docker run --env-file .env -it html-parser_local python -m cli.run_parser --s3 s3://data-pipeline-a64047a/runs/09-20-2022_20:26___b345453a-2166-4677-898e-78fc81f4f93a/loader_output/ s3://data-pipeline-a64047a/runs/test_output/

run_s3: 
	docker build -t html-parser-s3 . 
	docker run html-parser-s3 