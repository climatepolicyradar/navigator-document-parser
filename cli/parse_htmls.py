from typing import List, Union
from pathlib import Path
import logging
import logging.config

from tqdm import tqdm
from cloudpathlib import CloudPath

import sys

sys.path.append("..")

from src.base import ParserInput, ParserOutput  # noqa: E402
from src.html_parser.combined import CombinedParser  # noqa: E402


class TqdmLoggingHandler(logging.Handler):
    """Handler for logging to tqdm"""

    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        """Emit a log message"""
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception as e:
            print(f"Error emitting tqdm logging handler: {e}")
            self.handleError(record)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(TqdmLoggingHandler())


def copy_input_to_output_html(
    task: ParserInput, output_path: Union[Path, CloudPath]
) -> None:
    """Necessary to copy the input file to the output to ensure that we don't drop documents.

    The file is copied at the time of processing rather than syncing the entire input directory so that if that
    parser fails and retries it will not think that all files have already been parsed. :param task: input task
    specifying the document to copy :param output_path: path to save the copied file
    """
    blank_output = ParserOutput.parse_obj(
        {
            "document_id": task.document_id,
            "document_metadata": task.document_metadata,
            "document_name": task.document_name,
            "document_description": task.document_description,
            "document_url": task.document_url,
            "document_slug": task.document_slug,
            "document_content_type": task.document_content_type,
            "languages": None,
            "translated": "false",
            "html_data": {
                "text_blocks": [],
                "detected_date": None,
                "detected_title": "",
                "has_valid_text": False,
            },
            "pdf_data": None,
        }
    )

    output_path.write_text(blank_output.json(indent=4, ensure_ascii=False))

    logger.info(f"Blank output for {task.document_id} saved to {output_path}.")


def run_html_parser(input_tasks: List[ParserInput], output_dir: Union[Path, CloudPath]):
    """
    Run the parser on a list of input tasks specifying documents to parse, and save the results to an output directory.

    :param input_tasks: list of input tasks specifying documents to parse
    :param output_dir: directory of output JSON files (results)
    """

    logger.info("Running HTML parser")
    html_parser = CombinedParser()

    for task in tqdm(input_tasks):
        # TODO: validate the language detection probability threshold
        output_path = output_dir / f"{task.document_id}.json"

        copy_input_to_output_html(task, output_path)

        parsed_html = html_parser.parse(task).detect_and_set_languages()

        output_path.write_text(parsed_html.json(indent=4, ensure_ascii=False))

        logger.info(f"Output for {task.document_id} saved to {output_path}")
