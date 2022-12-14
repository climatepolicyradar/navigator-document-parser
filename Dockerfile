FROM python:3.9

RUN mkdir /app
WORKDIR /app

# Install dependencies for pdf2image and tesseract
RUN apt-get update
RUN apt-get install -y ffmpeg libsm6 libxext6 poppler-utils tesseract-ocr libtesseract-dev git

# Install pip and poetry
RUN pip install --upgrade pip
RUN pip install "poetry==1.2.2"

# Create layer for dependencies
COPY ./poetry.lock ./pyproject.toml ./

# Install python dependencies using poetry
RUN poetry config virtualenvs.create false
RUN poetry install
RUN pip install "git+https://github.com/facebookresearch/detectron2.git@v0.5#egg=detectron2"
RUN playwright install
RUN playwright install-deps

# Copy files to image
COPY ./data ./data
COPY ./src ./src
COPY ./cli ./cli
COPY ./.git ./.git
COPY ./.pre-commit-config.yaml ./.flake8 ./.gitignore ./

# Run the parser on the input s3 directory
CMD [ "sh", "./cli/run.sh" ]
