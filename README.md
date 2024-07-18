# Navigator Document Parser

## Overview

This repo contains a CLI application used for extracting text from pdf and html documents before translating these to a target language (default is English) if they are in a different language to the target language.

**HTML Text Extraction:**
- HTML webpages are processed by making a request to the webpage and extracting text from the html content using a combination of the `news-please` and `readability` python packages.

**PDF Text Extraction:**
- PDF documents are processed by downloading the pdf from the cdn (Content Delivery Network accessible via an endpoint) and using the `Azure` form recognizer API to extract text from the pdf.

**Translation:**
- Text is translated using the `Google` translation api.

## Setup

To operate and run the CLI the repo provides useful commands in the `Makefile`. This reads environment variables from a `.env` file. Create this locally by running the following command and then enter the relevant values.

``` bash
make setup
```

Once this is done we can then run the commands in the "Makefile". These split into two main groups, running directly on your machine, or in a Docker container.

To run locally run the following commands to install dependencies using Poetry and set up playwright and pre-commit. Then run the CLI locally.

Note that you will need a Python version in your virtual environment of that matches the project version. It is also recommended to run within a virtual environment.

``` bash
make install
make run_local
```

## Running the CLI

To run in Docker with the `data/raw` folder as input and `data/processed` as output run:

``` bash
make run_docker
```

The CLI operates on an input folder of tasks defined by JSON files in the following format as defined in `cpr_data_access` library dependency. This can be found [here](https://github.com/climatepolicyradar/data-access).

``` python
class ParserInput(BaseModel):
    """Base class for input to a parser."""

    document_id: str
    document_metadata: BackendDocument
    document_name: str
    document_description: str
    document_source_url: Optional[AnyHttpUrl]
    document_cdn_object: Optional[str]
    document_content_type: Optional[str]
    document_md5_sum: Optional[str]
    document_slug: str
```

It outputs JSON files named `${id}.json`, with `id` being the `document_id` of each input document, to an output folder in one of these formats:

``` python
class ParserOutput(BaseModel):
    """Base class for an output to a parser."""

    document_id: str
    document_metadata: BackendDocument
    document_name: str
    document_description: str
    document_source_url: Optional[AnyHttpUrl]
    document_cdn_object: Optional[str]
    document_content_type: Optional[str]
    document_md5_sum: Optional[str]
    document_slug: str

    languages: Optional[Sequence[str]] = None
    translated: bool = False
    html_data: Optional[HTMLData] = None
    pdf_data: Optional[PDFData] = None
```
