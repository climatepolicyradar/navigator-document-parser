import logging
import sys
from pathlib import Path
from typing import List, Union

import cloudpathlib.exceptions
from cloudpathlib import CloudPath
from tqdm.auto import tqdm

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
            document_metadata=task.document_metadata,
            document_name=task.document_name,
            document_description=task.document_description,
            document_source_url=task.document_source_url,
            document_cdn_object=task.document_cdn_object,
            document_md5_sum=task.document_md5_sum,
            languages=None,
            translated=False,
            document_slug=task.document_slug,
            document_content_type=None,
            html_data=None,
            pdf_data=None,
        )

        output_path = output_dir / f"{task.document_id}.json"
        try:
            output_path.write_text(output.json(indent=4, ensure_ascii=False))
        except cloudpathlib.exceptions.OverwriteNewerCloudError:
            logger.info(
                f"Tried to write {task.document_id} to {output_path}, received OverwriteNewerCloudError and "
                f"therefore skipping."
            )

        logger.info(f"Output for {task.document_id} saved to {output_path}")
