FROM python:3.9-slim-bullseye

RUN mkdir /app
WORKDIR /app

# Install pip and poetry
RUN pip install --upgrade pip
RUN pip install "poetry==1.5.1"

# Create layer for dependencies
COPY ./poetry.lock ./pyproject.toml ./

# Install python dependencies using poetry
RUN poetry config virtualenvs.create false
RUN poetry install
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

# Run the parser on the input s3 directory
CMD [ "sh", "./cli/run.sh" ]
