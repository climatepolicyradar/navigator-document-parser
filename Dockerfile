FROM python:3.9-slim-bullseye

RUN mkdir /app
WORKDIR /app

# Install dependencies for pdf2image and tesseract
RUN apt-get update
RUN apt-get install -y ffmpeg libsm6 libxext6 poppler-utils tesseract-ocr libtesseract-dev git pre-commit build-essential

# Install pip and poetry
RUN pip install --upgrade pip
RUN pip install "poetry==1.5.1"

# Create layer for dependencies
COPY ./poetry.lock ./pyproject.toml ./

# Install python dependencies using poetry
RUN poetry config virtualenvs.create false
RUN poetry export --with dev > requirements.txt
RUN pip3 install --no-cache -r requirements.txt
RUN pip3 install --no-cache "git+https://github.com/facebookresearch/detectron2.git@v0.5#egg=detectron2"
RUN playwright install
RUN playwright install-deps

# Copy files to image
COPY ./data ./data
COPY ./src ./src
COPY ./cli ./cli
COPY ./.git ./.git
COPY ./.pre-commit-config.yaml ./.flake8 ./.gitignore ./

# Pre-download the model
ENV PYTHONPATH "${PYTHONPATH}:/app"
RUN python3 '/app/cli/warm_up_model.py'

# Run the parser on the input s3 directory
CMD [ "sh", "./cli/run.sh" ]
