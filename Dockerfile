FROM python:3.9

RUN mkdir /app
WORKDIR /app

# Install dependencies for pdf2image and tesseract
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install poppler-utils -y
RUN apt-get install -y tesseract-ocr
RUN apt-get install -y libtesseract-dev

# Install pip and poetry
RUN pip install --upgrade pip
RUN pip install "poetry==1.1.8"

# Create layer for dependencies
COPY ./poetry.lock ./pyproject.toml ./

# Install python dependencies using poetry
RUN poetry config virtualenvs.create false
RUN poetry install
RUN pip install "git+https://github.com/facebookresearch/detectron2.git@v0.5#egg=detectron2"
RUN playwright install
RUN playwright install-deps

# Copy files to image
COPY ./src ./src
COPY ./cli ./cli
COPY ./data ./data
COPY ./credentials/google-creds.json ./credentials/google-creds.json

# Set environment variables
ARG GOOGLE_APPLICATION_CREDENTIALS=./credentials/google-creds.json
ENV GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS

ARG AWS_ACCESS_KEY_ID=
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID

ARG AWS_SECRET_ACCESS_KEY=
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY

ARG HTML_MIN_NO_LINES_FOR_VALID_TEXT=6
ENV HTML_MIN_NO_LINES_FOR_VALID_TEXT=$HTML_MIN_NO_LINES_FOR_VALID_TEXT

ARG HTML_HTTP_REQUEST_TIMEOUT=30
ENV HTML_HTTP_REQUEST_TIMEOUT=$HTML_HTTP_REQUEST_TIMEOUT

ARG HTML_MAX_PARAGRAPH_LENGTH_WORDS=500
ENV HTML_MAX_PARAGRAPH_LENGTH_WORDS=$HTML_MAX_PARAGRAPH_LENGTH_WORDS

ARG TARGET_LANGUAGES=en
ENV TARGET_LANGUAGES=$TARGET_LANGUAGES

ARG LAYOUTPARSER_MODEL_THRESHOLD_RESTRICTIVE=0.5
ENV LAYOUTPARSER_MODEL_THRESHOLD_RESTRICTIVE=$LAYOUTPARSER_MODEL_THRESHOLD_RESTRICTIVE

ARG LAYOUTPARSER_MODEL=mask_rcnn_X_101_32x8d_FPN_3x
ENV LAYOUTPARSER_MODEL=$LAYOUTPARSER_MODEL

ARG PDF_OCR_AGENT=tesseract
ENV PDF_OCR_AGENT=$PDF_OCR_AGENT

ARG CDN_DOMAIN=cdn.climatepolicyradar.org
ENV CDN_DOMAIN=$CDN_DOMAIN

# Run configuration 
ARG TEST_RUN=false
ENV TEST_RUN=$TEST_RUN

ARG RUN_PDF_PARSER=true
ENV RUN_PDF_PARSER=$RUN_PDF_PARSER

ARG RUN_HTML_PARSER=true
ENV RUN_HTML_PARSER=$RUN_HTML_PARSER


# Run the parser on the input s3 directory
CMD [ "sh", "./cli/run.sh" ]
