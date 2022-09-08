.PHONY: install test

install:
	poetry install
	poetry run pre-commit install
	poetry run playwright install 

test:
	python -m pytest