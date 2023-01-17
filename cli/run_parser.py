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

sys.path.append("..")

from src.base import (  # noqa: E402
    CONTENT_TYPE_HTML,
    CONTENT_TYPE_PDF,
    ParserInput,
)
from src.config import (  # noqa: E402
    FILES_TO_PARSE,
    RUN_HTML_PARSER,
    RUN_PDF_PARSER,
    RUN_TRANSLATION,
    TARGET_LANGUAGES,
    TEST_RUN,
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
        logger.info(f"FILESTOPARSE: {FILES_TO_PARSE}")
        env_files = FILES_TO_PARSE.split("$")[1:]

    files_to_parse: list[str] = list(files or [])
    files_to_parse.extend(env_files)

    if files_to_parse:
        logger.info(f"Only parsing files: {files_to_parse}")
    else:
        logger.info("Parsing all files")

    return list(
        (input_dir_as_path / f for f in files_to_parse)
        if files_to_parse
        else input_dir_as_path.glob("*.json")
    )  # type: ignore


@click.command()
@click.argument("input_dir", type=str)
@click.argument("output_dir", type=str)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu"]),
    help="Device to use for PDF parsing",
    required=True,
    default="cpu",
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
    help="Pass in a list of filenames to parse, relative to the input directory. Used to optionally specify a subset of files to parse.",
    multiple=True,
)
@click.option(
    "--redo",
    "-r",
    help="Redo parsing for files that have already been parsed. By default, files with IDs that already exist in the output directory are skipped.",
    is_flag=True,
    default=False,
)
@click.option(
    "--s3",
    help="Input and output directories are S3 paths. The CLI will download tasks from S3, run parsing, and upload the results to S3.",
    is_flag=True,
    default=False,
)
@click.option(
    "--debug", help="Run the parser with visual debugging", is_flag=True, default=False
)
def main(
    input_dir: str,
    output_dir: str,
    parallel: bool,
    device: str,
    files: Optional[tuple[str]],
    redo: bool,
    s3: bool,
    debug: bool,
):
    """
    Run the parser on a directory of JSON files specifying documents to parse, and save the results to an output directory.

    :param input_dir: directory of input JSON files (task specifications)
    :param output_dir: directory of output JSON files (results)
    :param parallel: whether to run PDF parsing over multiple processes
    :param device: device to use for PDF parsing
    :param files: list of filenames to parse, relative to the input directory. Can be used to select a subset of files to parse.
    :param redo: redo parsing for files that have already been parsed. Defaults to False.
    :param s3: input and output directories are S3 paths. The CLI will download tasks from S3, run parsing, and upload the results to S3.
    :param debug: whether to run in debug mode (save images of intermediate steps). Defaults to False.
    """

    if s3:
        input_dir_as_path = S3Path(input_dir)
        output_dir_as_path = S3Path(output_dir)
    else:
        input_dir_as_path = Path(input_dir)
        output_dir_as_path = Path(output_dir)

    # if visual debugging is on, create a debug directory
    if debug:
        debug_dir = output_dir_as_path / "debug"
        debug_dir.mkdir(exist_ok=True)  # type: ignore

    files_to_parse = _get_files_to_parse(files, input_dir_as_path)

    logger.info(
        "Run configuration.",
        extra={
            "props": {
                "Test Run": TEST_RUN,
                "Run PDF Parser": RUN_PDF_PARSER,
                "Run HTML Parser": RUN_HTML_PARSER,
            }
        },
    )

    tasks = []
    counter = 0
    for path in files_to_parse:
        if TEST_RUN and counter > 100:
            break
        else:
            try:
                tasks.append(ParserInput.parse_raw(path.read_text()))  # type: ignore
            except (pydantic.error_wrappers.ValidationError, KeyError) as e:
                logger.error(
                    "Failed to parse input file.",
                    extra={
                        "props": {
                            "Error Message": e,
                            "Document Path": str(path),
                        }
                    },
                )
        counter += 1

    # TODO: Update splitting to be based on ContentType enum
    no_processing_tasks = []
    html_tasks = []
    pdf_tasks = []
    output_tasks_paths = []
    for task in tasks:
        output_tasks_paths.append(output_dir_as_path / f"{task.document_id}.json")
        if task.document_content_type == CONTENT_TYPE_HTML:
            html_tasks.append(task)
        elif task.document_content_type == CONTENT_TYPE_PDF:
            pdf_tasks.append(task)
        else:
            no_processing_tasks.append(task)

    logger.info(
        "Tasks to process identified.",
        extra={
            "props": {
                "Total Tasks": len(tasks),
                "No Supported Content-Type Tasks": len(no_processing_tasks),
                "HTML Tasks": len(html_tasks),
                "PDF Tasks": len(pdf_tasks),
            }
        },
    )

    logger.info(
        f"Generating outputs for {len(no_processing_tasks)} inputs that cannot be processed."
    )
    process_documents_with_no_content_type(no_processing_tasks, output_dir_as_path)

    if RUN_HTML_PARSER:
        logger.info(f"Running HTML parser on {len(html_tasks)} documents.")
        run_html_parser(
            html_tasks,
            output_dir_as_path,
            redo=redo,
        )

    if RUN_PDF_PARSER:
        logger.info(f"Running PDF parser on {len(pdf_tasks)} documents.")
        run_pdf_parser(
            pdf_tasks,
            output_dir_as_path,
            parallel=parallel,
            device=device,
            debug=debug,
            redo=redo,
        )

    if RUN_TRANSLATION:
        logger.info(
            "Translating results to target languages specified in environment variables.",
            extra={
                "props": {
                    "Target Languages": ",".join(TARGET_LANGUAGES),
                }
            },
        )
        translate_parser_outputs(output_tasks_paths, redo=redo)


if __name__ == "__main__":
    main()
