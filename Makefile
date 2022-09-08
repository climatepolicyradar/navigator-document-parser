.PHONY: install test

install:
	poetry install
	poetry run pre-commit install
	poetry run playwright install 
	cp .env.example .env

test:
	python -m pytest