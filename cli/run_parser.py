import os
import logging
import logging.config
import json_logging
import sys
from pathlib import Path
from typing import Optional, Union

import click
import pydantic
from cloudpathlib import S3Path, CloudPath
from cpr_sdk.parser_models import (
    ParserInput,
    CONTENT_TYPE_HTML,
)

sys.path.append("..")

from src.config import (  # noqa: E402
    FILES_TO_PARSE,
    RUN_HTML_PARSER,
    RUN_PDF_PARSER,
    RUN_TRANSLATION,
    TARGET_LANGUAGES,
)
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


def _get_files_to_parse(
    files: Optional[tuple[str]],
    input_dir_as_path: Union[CloudPath, Path],
) -> list[Path]:
    # If no file list is provided, run over all inputs in the input prefix
    env_files = []
    if FILES_TO_PARSE is not None:
        _LOGGER.info(f"FILESTOPARSE: {FILES_TO_PARSE}")
        env_files = FILES_TO_PARSE.split("$")[1:]

    files_to_parse: list[str] = list(files or [])
    files_to_parse.extend(env_files)

    if files_to_parse:
        _LOGGER.info(f"Only parsing files: {files_to_parse}")
    else:
        _LOGGER.info("Parsing all files")

    return list(
        (input_dir_as_path / f for f in files_to_parse)
        if files_to_parse
        else input_dir_as_path.glob("*.json")
    )  # type: ignore


@click.command()
@click.argument("input_dir", type=str)
@click.argument("output_dir", type=str)
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
    "--files",
    "-f",
    help="Pass in a list of filenames to parse, relative to the input directory. Used "
    "to optionally specify a subset of files to parse.",
    multiple=True,
)
@click.option(
    "--redo",
    "-r",
    help="Redo parsing for files that have already been parsed. By default, files with "
    "IDs that already exist in the output directory are skipped.",
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
@click.option(
    "--debug", help="Run the parser with visual debugging", is_flag=True, default=False
)
def main(
    input_dir: str,
    output_dir: str,
    azure_api_response_cache_dir: str,
    parallel: bool,
    files: Optional[tuple[str]],
    redo: bool,
    s3: bool,
    debug: bool,
):
    """
    Run the parser on a directory of JSON files specifying documents to parse.

    Then save the results to an output directory.

    :param input_dir: directory of input JSON files (task specifications)
    :param output_dir: directory of output JSON files (results)
    :param azure_api_response_cache_dir: directory to store raw responses from Azure API during
        pdf parsing.
    :param parallel: whether to run PDF parsing over multiple processes
    :param files: list of filenames to parse, relative to the input directory.
        Can be used to select a subset of files to parse.
    :param redo: redo parsing for files that have already been parsed. Defaults to False.
    :param s3: input and output directories are S3 paths.
        The CLI will download tasks from S3, run parsing, and upload the results to S3.
    :param debug: whether to run in debug mode (save images of intermediate steps).
        Defaults to False.
    """

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

    # if visual debugging is on, create a debug directory
    if debug:
        debug_dir = output_dir_as_path / "debug"
        debug_dir.mkdir(exist_ok=True)  # type: ignore

    files_to_parse = _get_files_to_parse(files, input_dir_as_path)

    _LOGGER.info(
        "Run configuration.",
        extra={
            "props": {
                "run_pdf_parser": RUN_PDF_PARSER,
                "run_html_parser": RUN_HTML_PARSER,
            }
        },
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
                "html_asks": len(html_tasks),
                "pdf_tasks": len(pdf_tasks),
            }
        },
    )

    _LOGGER.info(
        f"Generating outputs for {len(no_processing_tasks)} inputs that cannot be "
        f"processed. "
    )
    process_documents_with_no_content_type(no_processing_tasks, output_dir_as_path)

    if RUN_HTML_PARSER:
        _LOGGER.info(f"Running HTML parser on {len(html_tasks)} documents.")
        run_html_parser(
            html_tasks,
            output_dir_as_path,
            redo=redo,
        )

    if RUN_PDF_PARSER:
        _LOGGER.info(f"Running PDF parser on {len(pdf_tasks)} documents.")
        run_pdf_parser(
            pdf_tasks,
            output_dir_as_path,
            azure_cache_dir_as_path,
            parallel=parallel,
            debug=debug,
            redo=redo,
        )

    if RUN_TRANSLATION:
        _LOGGER.info(
            "Translating results to target languages specified in env variables.",
            extra={
                "props": {
                    "target_languages": ",".join(TARGET_LANGUAGES),
                }
            },
        )
        translate_parser_outputs(output_tasks_paths, redo=redo)


if __name__ == "__main__":
    main()
