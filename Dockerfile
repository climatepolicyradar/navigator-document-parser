FROM python:3.9

RUN mkdir /app
WORKDIR /app

# Install pip and poetry
RUN pip install --upgrade pip
RUN pip install "poetry==1.1.8"

# Create layer for dependencies
COPY ./poetry.lock ./pyproject.toml ./

# Install python dependencies using poetry
RUN poetry config virtualenvs.create false
RUN poetry install

# Copy files to image
COPY ./src ./src
COPY ./cli ./cli
