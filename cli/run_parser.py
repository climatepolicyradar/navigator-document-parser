from pathlib import Path
import os
import logging
import logging.config

import click

import sys

sys.path.append("..")

from src.base import ParserInput  # noqa: E402
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
@click.argument(
    "input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.argument(
    "output_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
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
def main(input_dir: Path, output_dir: Path, parallel: bool, device: str):
    """
    Run the parser on a directory of JSON files specifying documents to parse, and save the results to an output directory.

    :param input_dir: directory of input JSON files (task specifications)
    :param output_dir: directory of output JSON files (results)
    """

    tasks = [ParserInput.parse_file(_path) for _path in input_dir.glob("*.json")]

    html_tasks = [task for task in tasks if task.content_type == "text/html"]
    pdf_tasks = [task for task in tasks if task.content_type == "application/pdf"]

    logger.info(f"Fount {len(html_tasks)} HTML tasks and {len(pdf_tasks)} PDF tasks")

    logger.info(f"Running HTML parser on {len(html_tasks)} documents")
    run_html_parser(html_tasks, output_dir)

    logger.info(f"Running PDF parser on {len(pdf_tasks)} documents")
    run_pdf_parser(pdf_tasks, output_dir, parallel=parallel, device=device)


if __name__ == "__main__":
    main()
