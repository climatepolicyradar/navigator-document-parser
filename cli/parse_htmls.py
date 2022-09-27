from typing import List, Union
from pathlib import Path
import logging
import logging.config

from tqdm.auto import tqdm
from cloudpathlib import CloudPath

import sys

sys.path.append("..")

from src.base import ParserInput  # noqa: E402
from src.html_parser.combined import CombinedParser  # noqa: E402


logger = logging.getLogger(__name__)


def run_html_parser(input_tasks: List[ParserInput], output_dir: Union[Path, CloudPath]):
    """
    Run the parser on a list of input tasks specifying documents to parse, and save the results to an output directory.

    :param input_dir: directory of input JSON files (task specifications)
    :param output_dir: directory of output JSON files (results)
    """

    logger.info("Running HTML parser")

    for task in tqdm(input_tasks):
        html_parser = CombinedParser()
        # TODO: validate the language detection probability threshold
        parsed_html = html_parser.parse(task).detect_and_set_languages()
        output_path = output_dir / f"{task.id}.json"

        output_path.write_text(parsed_html.json(indent=4, ensure_ascii=False))

        logger.info(f"Output for {task.id} saved to {output_path}")
