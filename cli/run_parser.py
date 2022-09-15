from pathlib import Path
import os
import logging
import logging.config
from typing import List, Optional
import sys

import click
from cloudpathlib import S3Path

sys.path.append("..")

from src.base import ParserInput, ParserOutput  # noqa: E402
from cli.parse_htmls import run_html_parser  # noqa: E402
from cli.parse_pdfs import run_pdf_parser  # noqa: E402

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
def main(
    input_dir: str,
    output_dir: str,
    parallel: bool,
    device: str,
    files: Optional[List[str]],
    redo: bool,
    s3: bool,
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
    """

    if s3:
        input_dir_as_path = S3Path(input_dir)
        output_dir_as_path = S3Path(output_dir)
    else:
        input_dir_as_path = Path(input_dir)
        output_dir_as_path = Path(output_dir)

    # We use `parse_raw(path.read_text())` instead of `parse_file(path)` because the latter tries to coerce CloudPath objects to pathlib.Path objects.
    document_ids_previously_parsed = set(
        [
            ParserOutput.parse_raw(path.read_text()).id
            for path in output_dir_as_path.glob("*.json")
        ]
    )

    files_to_parse = (
        (input_dir_as_path / f for f in files)
        if files
        else input_dir_as_path.glob("*.json")
    )

    tasks = [ParserInput.parse_raw(path.read_text()) for path in files_to_parse]

    if not redo and document_ids_previously_parsed.intersection(
        {task.id for task in tasks}
    ):
        logger.warning(
            f"Found {len(document_ids_previously_parsed.intersection({task.id for task in tasks}))} documents that have already parsed. Skipping."
        )
        tasks = [
            task for task in tasks if task.id not in document_ids_previously_parsed
        ]

    html_tasks = [task for task in tasks if task.content_type == "text/html"]
    pdf_tasks = [task for task in tasks if task.content_type == "application/pdf"]

    logger.info(f"Found {len(html_tasks)} HTML tasks and {len(pdf_tasks)} PDF tasks")

    logger.info(f"Running HTML parser on {len(html_tasks)} documents")
    run_html_parser(html_tasks, output_dir_as_path)

    logger.info(f"Running PDF parser on {len(pdf_tasks)} documents")
    run_pdf_parser(pdf_tasks, output_dir_as_path, parallel=parallel, device=device)


if __name__ == "__main__":
    main()
