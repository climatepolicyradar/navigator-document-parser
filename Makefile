.PHONY: parse install test

parse:
	python -m cli.run_parser ./data/raw ./data/processed

install:
	poetry install
	poetry run pre-commit install
	poetry run playwright install 
	cp .env.example .env

test:
	python -m pytest

docker_build:
	docker build -t html-parser .

docker_test:
	docker run --network host html-parser python -m pytest