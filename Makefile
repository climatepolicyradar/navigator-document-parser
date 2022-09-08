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