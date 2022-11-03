import logging
import logging.config
import sys
from pathlib import Path
from typing import List, Union

import cloudpathlib.exceptions
from cloudpathlib import CloudPath
from tqdm import tqdm

sys.path.append("..")

from src.base import HTMLData, ParserInput, ParserOutput  # noqa: E402
from src.html_parser.combined import CombinedParser  # noqa: E402

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def copy_input_to_output_html(
        task: ParserInput, output_path: Union[Path, CloudPath]
) -> None:
    """Necessary to copy the input file to the output to ensure that we don't drop documents.

    The file is copied at the time of processing rather than syncing the entire input directory so that if that
    parser fails and retries it will not think that all files have already been parsed. :param task: input task
    specifying the document to copy :param output_path: path to save the copied file
    """
    blank_output = ParserOutput(
        document_id=task.document_id,
        document_metadata=task.document_metadata,
        document_name=task.document_name,
        document_description=task.document_description,
        document_source_url=task.document_source_url,
        document_cdn_object=task.document_cdn_object,
        document_md5_sum=task.document_md5_sum,
        document_slug=task.document_slug,
        document_content_type=task.document_content_type,
        languages=None,
        translated=False,
        html_data=HTMLData(
            text_blocks=[],
            detected_date=None,
            detected_title="",
            has_valid_text=False,
        ),
        pdf_data=None,
    )

    output_path.write_text(blank_output.json(indent=4, ensure_ascii=False))

    logger.info(f"Blank output for {task.document_id} saved to {output_path}.")


def run_html_parser(input_tasks: List[ParserInput], output_dir: Union[Path, CloudPath], redo: bool = False):
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

        if not output_path.exists():  # type: ignore
            copy_input_to_output_html(task, output_path)  # type: ignore

        existing_parser_output = ParserOutput.parse_raw(output_path.read_text())  # type: ignore
        # If no parsed html dta exists, assume we've not run before
        existing_html_data_exists = (
            existing_parser_output.html_data is not None and
            existing_parser_output.html_data.text_blocks
        )
        should_run_parser = not existing_html_data_exists or redo
        if not should_run_parser:
            logger.info(f"Skipping already parsed html with output - {output_path}.")
            continue

        parsed_html = html_parser.parse(task).detect_and_set_languages()

        try:
            output_path.write_text(parsed_html.json(indent=4, ensure_ascii=False))  # type: ignore
        except cloudpathlib.exceptions.OverwriteNewerCloudError:
            logger.info(f"Tried to write {task.document_id} to {output_path}, received OverwriteNewerCloudError and "
                        f"therefore skipping.")

        logger.info(f"Output for {task.document_id} saved to {output_path}")
