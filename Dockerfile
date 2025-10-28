FROM python:3.10-slim-bullseye

RUN mkdir /app
WORKDIR /app

# Copy the src module here so that poetry can install it as a package
COPY ./src ./src

# Install git and precommit
RUN apt-get update
RUN apt-get install -y git pre-commit

# Install pip and poetry
RUN pip install --upgrade pip
RUN pip install "poetry==1.8.3"

# Create layer for dependencies
COPY ./poetry.lock ./pyproject.toml ./

# Install python dependencies using poetry
RUN poetry config virtualenvs.create false
RUN poetry install
RUN playwright install
RUN playwright install-deps

# Copy files to image
COPY ./data ./data
COPY ./cli ./cli
COPY ./.git ./.git
COPY ./.pre-commit-config.yaml ./.flake8 ./.gitignore ./

# Add the app directory to the PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Run the parser on the input s3 directory
ENTRYPOINT ["python3", "-m", "cli.run_parser"]
