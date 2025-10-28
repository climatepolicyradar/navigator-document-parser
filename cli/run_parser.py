import os
import logging
import logging.config
import json_logging
import sys
from pathlib import Path

import click
import base64
import pydantic
from cloudpathlib import CloudPath, S3Path
from cpr_sdk.parser_models import CONTENT_TYPE_HTML, ParserInput
from typing import NewType

sys.path.append("..")

from src.config import TARGET_LANGUAGES  # noqa: E402
from cli.parse_htmls import run_html_parser  # noqa: E402
from cli.parse_pdfs import run_pdf_parser  # noqa: E402
from cli.parse_no_content_type import (  # noqa: E402
    process_documents_with_no_content_type,
)
from cli.translate_outputs import translate_parser_outputs  # noqa: E402

# Clear existing log handlers so we always log in structured JSON
root_logger = logging.getLogger()
if root_logger.handlers:
    for handler in root_logger.handlers:
        root_logger.removeHandler(handler)

for _, logger in logging.root.manager.loggerDict.items():
    if isinstance(logger, logging.Logger):
        logger.propagate = True
        if logger.handlers:
            for handler in logger.handlers:
                logger.removeHandler(handler)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

DEFAULT_LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "default": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",  # Default is stderr
        },
    },
    "loggers": {},
    "root": {
        "handlers": ["default"],
        "level": LOG_LEVEL,
    },
}
logging.config.dictConfig(DEFAULT_LOGGING)
json_logging.init_non_web(enable_json=True)
_LOGGER = logging.getLogger(__name__)

# Example: CCLW.executive.1813.2418
DocumentImportId = NewType("DocumentImportId", str)


class CommaSeparatedList(click.ParamType):
    """A Custom ParamType allowing comma separated lists to be pass to the cli."""

    name = "comma_separated_list"

    def convert(self, value, param, ctx):
        """Convert the value passed in to the cli to the desired format."""
        if value is None:
            return None
        return [item.strip() for item in value.split(",") if item.strip()]


def setup_google_credentials() -> None:
    """Setup a local credentials file for use by the Google Translation API Client"""

    # Create credentials directory
    credentials_dir = Path("/app/credentials")
    credentials_dir.mkdir(exist_ok=True)

    # Decode base64 and write to file
    google_creds_encoded: str = os.environ["GOOGLE_CREDS"]
    google_creds_decoded = base64.b64decode(google_creds_encoded)
    creds_file = credentials_dir / "google-creds.json"
    creds_file.write_bytes(google_creds_decoded)

    # Set environment variable
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_file)


@click.command()
@click.argument("input_dir", type=str)
@click.argument("output_dir", type=str)
@click.argument("document_import_ids", type=CommaSeparatedList())
@click.option(
    "--azure_api_response_cache_dir",
    help="Directory to store raw responses from Azure API during pdf parsing.",
    nargs=1,
    type=str,
    default=None,
    required=False,
)
@click.option(
    "--parallel",
    help="Whether to run PDF parsing over multiple processes",
    is_flag=True,
    default=False,
)
@click.option(
    "--s3",
    help="Input and output directories are S3 paths. The CLI will download tasks from "
    "S3, run parsing, and upload the results to S3.",
    is_flag=True,
    default=False,
)
def main(
    input_dir: str,
    output_dir: str,
    document_import_ids: list[DocumentImportId],
    azure_api_response_cache_dir: str,
    parallel: bool,
    s3: bool,
):
    """
    Run the parser on a directory of JSON files specifying documents to parse.

    Then save the results to an output directory.

    :param input_dir: directory of input JSON files (task specifications)
    :param output_dir: directory of output JSON files (results)
    :param document_import_ids: Documents specified by Import Id to run parsing on.
    :param azure_api_response_cache_dir: directory to store raw responses from Azure API during
        pdf parsing.
    :param parallel: whether to run PDF parsing over multiple processes
    :param s3: input and output directories are S3 paths.
        The CLI will download tasks from S3, run parsing, and upload the results to S3.
    """

    setup_google_credentials()

    if s3:
        input_dir_as_path = S3Path(input_dir)
        output_dir_as_path = S3Path(output_dir)
        azure_cache_dir_as_path = (
            S3Path(azure_api_response_cache_dir)
            if azure_api_response_cache_dir
            else None
        )
    else:
        input_dir_as_path = Path(input_dir)
        output_dir_as_path = Path(output_dir)
        azure_cache_dir_as_path = (
            Path(azure_api_response_cache_dir) if azure_api_response_cache_dir else None
        )

    files_to_parse: set[CloudPath | Path] = set(
        (input_dir_as_path / f"{f}.json" for f in document_import_ids)
    )

    tasks = []
    for path in files_to_parse:
        try:
            tasks.append(ParserInput.model_validate_json(path.read_text()))
        except (pydantic.ValidationError, KeyError) as e:
            _LOGGER.error(
                "Failed to parse input file.",
                extra={
                    "props": {
                        "error_message": e,
                        "document_path": str(path),
                    }
                },
            )

    # TODO: Update splitting to be based on ContentType enum
    no_processing_tasks = []
    html_tasks = []
    pdf_tasks = []
    output_tasks_paths = []
    for task in tasks:
        output_tasks_paths.append(output_dir_as_path / f"{task.document_id}.json")
        if (
            task.document_cdn_object is not None
            and task.document_cdn_object.lower().endswith(".pdf")
        ):
            pdf_tasks.append(task)
        elif task.document_content_type == CONTENT_TYPE_HTML:
            # This code path should never be hit as we convert all HTML to PDF
            html_tasks.append(task)
        else:
            no_processing_tasks.append(task)

    _LOGGER.info(
        "Tasks to process identified.",
        extra={
            "props": {
                "total_tasks": len(tasks),
                "no_supported _content-type_tasks": len(no_processing_tasks),
                "html_tasks": len(html_tasks),
                "pdf_tasks": len(pdf_tasks),
            }
        },
    )

    _LOGGER.info(
        f"Generating outputs for {len(no_processing_tasks)} inputs that cannot be "
        f"processed. "
    )
    process_documents_with_no_content_type(no_processing_tasks, output_dir_as_path)

    _LOGGER.info(f"Running HTML parser on {len(html_tasks)} documents.")
    run_html_parser(
        html_tasks,
        output_dir_as_path,
    )

    _LOGGER.info(f"Running PDF parser on {len(pdf_tasks)} documents.")
    run_pdf_parser(
        pdf_tasks,
        output_dir_as_path,
        azure_cache_dir_as_path,
        parallel=parallel,
    )

    _LOGGER.info(
        "Translating results to target languages specified in env variables.",
        extra={
            "props": {
                "target_languages": ",".join(TARGET_LANGUAGES),
            }
        },
    )
    translate_parser_outputs(output_tasks_paths)


if __name__ == "__main__":
    main()
