from pathlib import Path
from typing import Union
import logging

from cloudpathlib import CloudPath
from tqdm.auto import tqdm

from src.config import TARGET_LANGUAGES  # noqa: E402
from src.base import ParserOutput  # noqa: E402
from src.translator.translate import translate_parser_output  # noqa: E402

logger = logging.getLogger(__name__)


def translate_parser_outputs(parser_output_dir: Union[Path, CloudPath]) -> None:
    """
    Translate parser outputs saved in the output directory, and save the translated outputs to the output directory.

    :param parser_output_dir: directory containing parser outputs
    """

    for path in tqdm(parser_output_dir.glob("*.json")):
        logger.debug(f"Translating {path}.")

        parser_output = ParserOutput.parse_raw(path.read_text())
        logger.debug(f"Successfuly parsed {path} for translation.")

        # Skip already translated outputs. Note this does not prevent the CLI from translating existing parser outputs again,
        # but instead makes sure it doesn't translate a translation.
        if parser_output.translated:
            logger.debug(f"parser.translated true for - {path}.")
            continue

        _target_languages = set(TARGET_LANGUAGES)

        # If there is only one language that's been detected in the parser output, we don't need to translate to this language.
        # TODO: how do we deal with the fact that parser outputs can contain multiple languages here?
        if parser_output.languages and len(parser_output.languages) == 1:
            _target_languages = _target_languages - set(parser_output.languages)

        logger.debug(f"Target languages for {path}: {_target_languages}.")

        for target_language in _target_languages:
            logger.debug(f"Translating {path} to {target_language}.")
            output_path = path.with_name(
                f"{path.stem}_translated_{target_language}.json"
            )
            if output_path.exists():
                logger.info(
                    f"Skipping translating {output_path} because it already exists."
                )
                continue

            translated_parser_output = translate_parser_output(
                parser_output, target_language
            )
            logger.debug(f"Translated {path} to {target_language}.")

            output_path.write_text(
                translated_parser_output.json(indent=4, ensure_ascii=False)
            )
            logger.debug(f"Saved translated output to {output_path}.")
