.PHONY: data

data:
	python -m src.get_data ./data/raw/processed_policies.csv ./data/interim/html_sample.csv

run_parsers: data
	python run_parsers.py ./data/interim/html_sample.csv ./data/processed

install:
	pip install -r requirements.txt
	pre-commit install
	playwright install 