from pathlib import Path
import os
import logging
import logging.config
from typing import List, Optional
import sys

import click
from cloudpathlib import S3Path
import pydantic  # noqa: E402
from datetime import datetime

sys.path.append("..")

from src.base import ParserInput, ParserOutput, StandardErrorLog  # noqa: E402
from src.config import TARGET_LANGUAGES  # noqa: E402
from src.config import TEST_RUN  # noqa: E402
from src.config import RUN_PDF_PARSER  # noqa: E402
from src.config import RUN_HTML_PARSER  # noqa: E402
from src.config import FILES_TO_PARSE  # noqa: E402
from cli.parse_htmls import run_html_parser  # noqa: E402
from cli.parse_pdfs import run_pdf_parser  # noqa: E402
from cli.parse_no_content_type import (  # noqa: E402
    process_documents_with_no_content_type,
)
from cli.translate_outputs import translate_parser_outputs  # noqa: E402

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
DEFAULT_LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",  # Default is stderr
            "formatter": "json",
        },
    },
    "loggers": {},
    "root": {
        "handlers": ["console"],
        "level": LOG_LEVEL,
    },
    "formatters": {"json": {"()": "pythonjsonlogger.jsonlogger.JsonFormatter"}},
}

logger = logging.getLogger(__name__)
logging.config.dictConfig(DEFAULT_LOGGING)


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
    files: Optional[List[str]],
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
    # TODO use standard logging format for all logs.

    if s3:
        input_dir_as_path = S3Path(input_dir)
        output_dir_as_path = S3Path(output_dir)
    else:
        input_dir_as_path = Path(input_dir)
        output_dir_as_path = Path(output_dir)

    # if visual debugging is on, create a debug directory
    if debug:
        debug_dir = output_dir_as_path / "debug"
        debug_dir.mkdir(exist_ok=True)

    # We use `parse_raw(path.read_text())` instead of `parse_file(path)` because the latter tries to coerce CloudPath
    # objects to pathlib.Path objects.
    document_ids_previously_parsed = []
    for path in output_dir_as_path.glob("*.json"):
        try:
            document_ids_previously_parsed.append(
                ParserOutput.parse_raw(path.read_text()).document_id
            )
        except pydantic.ValidationError as e:
            logger.error(
                StandardErrorLog.parse_obj(
                    {
                        "timestamp": datetime.now(),
                        "pipeline_stage": "Parser: Parse output files in the output directory to find already parsed.",
                        "status_code": "None",
                        "error_type": "ParserOutputValidationError",
                        "message": f"{e}",
                        "document_in_process": path,
                    }
                )
            )
    document_ids_previously_parsed = set(document_ids_previously_parsed)

    # TODO clean this filth up
    if files:
        logger.info(f"Only parsing files: {files}")
    else:
        logger.info("Parsing all files")

    if FILES_TO_PARSE is not None:
        logger.info(f"FILESTOPARSE: {FILES_TO_PARSE}")
        files = FILES_TO_PARSE.split("$")[1:]

    if files:
        logger.info(f"Only parsing files: {files}")
    else:
        logger.info("Parsing all files")

    files_to_parse = (
        (input_dir_as_path / f for f in files)
        if files
        else input_dir_as_path.glob("*.json")
    )

    logger.info(
        f"Run configuration TEST_RUN:{TEST_RUN}, RUN_PDF_PARSER:{RUN_PDF_PARSER}, RUN_HTML_PARSER:{RUN_HTML_PARSER}"
    )

    tasks = []
    counter = 0
    for path in files_to_parse:
        if TEST_RUN and counter > 100:
            break
        else:
            try:
                tasks.append(ParserInput.parse_raw(path.read_text()))

            except pydantic.error_wrappers.ValidationError as e:
                logger.error(
                    StandardErrorLog.parse_obj(
                        {
                            "timestamp": datetime.now(),
                            "pipeline_stage": "Parser: Parse the input files in the input directory.",
                            "status_code": "None",
                            "error_type": "ParserInputValidationError",
                            "message": f"{e}",
                            "document_in_process": path,
                        }
                    )
                )
        counter += 1

    if not redo and document_ids_previously_parsed.intersection(
        {task.document_id for task in tasks}
    ):
        logger.warning(
            f"Found {len(document_ids_previously_parsed.intersection({task.document_id for task in tasks}))} documents that have already parsed. Skipping."
        )
        tasks = [
            task
            for task in tasks
            if task.document_id not in document_ids_previously_parsed
        ]

    no_document_tasks = [
        task for task in tasks if task.document_content_type is None
    ]  # tasks without a URL or content type
    html_tasks = [task for task in tasks if task.document_content_type == "text/html"]
    pdf_tasks = [
        task for task in tasks if task.document_content_type == "application/pdf"
    ]

    logger.info(
        f"Found {len(html_tasks)} HTML tasks, {len(pdf_tasks)} PDF tasks, and {len(no_document_tasks)} tasks without a document to parse."
    )

    logger.info(
        f"Generating outputs for {len(no_document_tasks)} inputs with URL or content type."
    )
    process_documents_with_no_content_type(no_document_tasks, output_dir_as_path)

    if RUN_HTML_PARSER:
        logger.info(f"Running HTML parser on {len(html_tasks)} documents")
        run_html_parser(html_tasks, output_dir_as_path)

    if RUN_PDF_PARSER:
        logger.info(f"Running PDF parser on {len(pdf_tasks)} documents")
        run_pdf_parser(
            pdf_tasks, output_dir_as_path, parallel=parallel, device=device, debug=debug
        )

    logger.info(
        f"Translating results to target languages specified in environment variables: {','.join(TARGET_LANGUAGES)}"
    )
    translate_parser_outputs(output_dir_as_path)


if __name__ == "__main__":
    main()
