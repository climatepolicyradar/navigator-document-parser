FROM python:3.9

RUN mkdir /app
WORKDIR /app

RUN apt-get clean
RUN apt-get update

ENV PYTHONDONTWRITEBYTECODE=1
ENV PLAYWRIGHT_BROWSERS_PATH=/app/ms-playwright
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip and poetry
RUN pip install --upgrade pip
RUN pip install "poetry==1.5.1"

# Create layer for dependencies
COPY ./poetry.lock ./pyproject.toml ./

# Install python dependencies using poetry
RUN poetry config virtualenvs.create false
RUN poetry install
RUN pip install "git+https://github.com/facebookresearch/detectron2.git@v0.5#egg=detectron2"

# Copy files to image
COPY ./data ./data
COPY ./src ./src
COPY ./cli ./cli
COPY ./.git ./.git
COPY ./.pre-commit-config.yaml ./.flake8 ./.gitignore ./

RUN pip install playwright
# install manually all the missing libraries
RUN apt-get install -y gconf-service libasound2 libatk1.0-0 libcairo2 libcups2 libfontconfig1 libgdk-pixbuf2.0-0 libgtk-3-0 libnspr4 libpango-1.0-0 libxss1 fonts-liberation libappindicator1 libnss3 lsb-release xdg-utils
RUN PLAYWRIGHT_BROWSERS_PATH=/app/ms-playwright python -m playwright install --with-deps chromium

# Pre-download the model
ENV PYTHONPATH "${PYTHONPATH}:/app"
RUN python '/app/cli/warm_up_model.py'

# Run the parser on the input s3 directory
CMD [ "sh", "./cli/run.sh" ]