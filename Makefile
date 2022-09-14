.PHONY: run_local run_docker install test_local build test

install:
	poetry install
	poetry run pre-commit install
	poetry run playwright install 
	cp .env.example .env

run_local:
	python -m cli.run_parser ./data/raw ./data/processed

test_local:
	LAYOUTPARSER_MODEL=faster_rcnn_R_50_FPN_3x OCR_AGENT=tesseract python -m pytest

build:
	docker build -t html-parser .

test:
	docker run --network host html-parser LAYOUTPARSER_MODEL=faster_rcnn_R_50_FPN_3x OCR_AGENT=tesseract python -m pytest -vvv

run_docker:
	docker run --network host -v ${PWD}/data:/app/data html-parser python -m cli.run_parser ./data/raw ./data/processed