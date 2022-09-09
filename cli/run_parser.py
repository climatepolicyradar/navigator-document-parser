from pathlib import Path
import os
import logging
import logging.config

import click
from tqdm.auto import tqdm

import sys

sys.path.append("..")

from src.base import HTMLParserInput  # noqa: E402
from src.combined import CombinedParser  # noqa: E402

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
def main(input_dir: Path, output_dir: Path):
    """
    Run the parser on a directory of JSON files specifying documents to parse, and save the results to an output directory.

    :param input_dir: directory of input JSON files (task specifications)
    :param output_dir: directory of output JSON files (results)
    """

    tasks = (HTMLParserInput.parse_file(_path) for _path in input_dir.glob("*.json"))

    html_parser = CombinedParser()

    logger.info("Running HTML parser")

    for task in tqdm(tasks):
        parsed_html = html_parser.parse(task).detect_language()
        output_path = output_dir / f"{task.id}.json"

        with open(output_path, "w") as f:
            f.write(parsed_html.json(indent=4, ensure_ascii=False))

        logger.info(f"Output for {task.id} saved to {output_path}")


if __name__ == "__main__":
    main()
