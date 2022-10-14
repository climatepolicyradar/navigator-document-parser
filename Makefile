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
	LAYOUTPARSER_MODEL=faster_rcnn_R_50_FPN_3x PDF_OCR_AGENT=tesseract TARGET_LANGUAGES=en python -m pytest -vvv

build:
	cp Dockerfile.local.example Dockerfile
	docker build -t html-parser .

test:
	docker run -e "LAYOUTPARSER_MODEL=faster_rcnn_R_50_FPN_3x" -e "PDF_OCR_AGENT=tesseract" --network host html-parser python -m pytest -vvv

run_docker:
	docker run --network host -v ${PWD}/data:/app/data html-parser python -m cli.run_parser ./data/raw ./data/processed

run_local_against_s3:
	cp Dockerfile.aws.example Dockerfile
	docker build -t html-parser_s3 .
	docker run -e s3_in=s3://data-pipeline-a64047a/unit_tests/runs/loader_output/ -e s3_out=s3://data-pipeline-a64047a/unit_tests/runs/parser_output/ -it html-parser_s3

build_and_push_ecr:
	cp Dockerfile.aws.example Dockerfile
	aws ecr get-login-password --region eu-west-2 --profile dev | docker login --username AWS --password-stdin 073457443605.dkr.ecr.eu-west-2.amazonaws.com
	docker build -t parser-staging-pipeline-b5871ba .
	docker tag parser-staging-pipeline-b5871ba:latest 073457443605.dkr.ecr.eu-west-2.amazonaws.com/parser-staging-pipeline-b5871ba:latest
	docker push 073457443605.dkr.ecr.eu-west-2.amazonaws.com/parser-staging-pipeline-b5871ba:latest