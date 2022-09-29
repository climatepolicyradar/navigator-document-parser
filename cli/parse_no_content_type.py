from typing import List, Union
from pathlib import Path
import logging
import sys

from tqdm.auto import tqdm
from cloudpathlib import CloudPath

sys.path.append("..")

from src.base import ParserInput, ParserOutput  # noqa: E402

logger = logging.getLogger(__name__)


def process_documents_with_no_content_type(
    input_tasks: List[ParserInput], output_dir: Union[Path, CloudPath]
):
    """
    Generates outputs with the same values as each input, and null values for the parsed content.

    :param input_tasks: list of input tasks specifying documents to parse
    :param output_dir: directory of output JSON files (results)
    """

    for task in tqdm(input_tasks):
        output = ParserOutput(
            document_id=task.document_id,
            document_name=task.document_name,
            document_description=task.document_description,
            document_url=task.document_url,
            languages=None,
            translated=False,
            document_slug=task.document_slug,
            document_content_type=None,
            html_data=None,
            pdf_data=None,
        )

        output_path = output_dir / f"{task.document_id}.json"
        output_path.write_text(output.json(indent=4, ensure_ascii=False))

        logger.info(f"Output for {task.document_id} saved to {output_path}")
