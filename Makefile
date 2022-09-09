.PHONY: parse install test_local build test

parse:
	python -m cli.run_parser ./data/raw ./data/processed

install:
	poetry install
	poetry run pre-commit install
	poetry run playwright install 
	cp .env.example .env

test_local:
	python -m pytest

build:
	docker build -t html-parser .

test:
	docker run --network host html-parser python -m pytest