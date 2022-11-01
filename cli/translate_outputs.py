from typing import Set, Sequence
import logging

import logging
from typing import Set, Sequence

from cloudpathlib import S3Path
from tqdm.auto import tqdm

from src.base import ParserOutput
from src.config import TARGET_LANGUAGES, LOGGING_LEVEL
from src.translator.translate import translate_parser_output

logger = logging.getLogger(__name__)
level = logging.getLevelName(LOGGING_LEVEL)
logger.setLevel(level)


def should_be_translated(document: ParserOutput) -> bool:
    """Determine if a document should be translated.

    If the document has not already been translated and has not null source url, then it should be translated."""
    if document.translated or document.document_source_url is None:
        return False
    return True


def identify_translation_languages(document: ParserOutput, target_languages: Set) -> Set:
    """Determine the languages to translate a document to.

    We subtract the current document languages from the target languages to determine the languages to translate to.
    E.g. doc.languages=['fr'] and target_languages=['en'] -> ['en'] - ['fr'] -> ['en'] (translate to English)
    E.g. doc.languages=['en'] and target_languages=['en'] -> ['en'] - ['en'] -> [] (no languages to translate to)

    If there are no detected document languages then we translate to all target languages.
    """
    # TODO: how do we deal with the fact that parser outputs can contain multiple languages here?
    if document.languages and len(document.languages) == 1:
        minus = set(document.languages)
        logger.debug(f"Removing {minus if minus is not None else None} from {target_languages}.")
        target_languages = target_languages - set(document.languages)
    return target_languages


def translate_parser_outputs(task_output_paths: Sequence[str]) -> None:
    """
    Translate parser outputs saved in the output directory, and save the translated outputs to the output directory.

    :param task_output_paths: A list of the paths to the parser outputs for this current instance to translate.
    """
    _target_languages = set(TARGET_LANGUAGES)

    for path in tqdm(task_output_paths):
        logger.debug(f"Translator processing - {path}.")

        try:
            parser_output = ParserOutput.parse_raw(path.read_text())
            logger.debug(f"Successfully parsed {path} from output dir during translation processing.")
        except FileNotFoundError:
            logger.error(f"Could not find {path} in output dir during translation processing.")
            continue

        if should_be_translated(parser_output):
            logger.debug(f"Document should be translated: {path}")

            target_languages = identify_translation_languages(parser_output, _target_languages)
            logger.debug(f"Target languages: {target_languages} for {path}")

            for target_language in target_languages:
                logger.debug(f"Translating {path} to {target_language}.")

                output_path = path.with_name(f"{path.stem}_translated_{target_language}.json")
                if output_path.exists():
                    logger.info(f"Skipping translating {output_path} because it already exists.")
                    continue

                translated_parser_output = translate_parser_output(
                    parser_output, target_language
                )
                logger.debug(f"Translated {path} to {target_language}.")

                try:
                    output_path.write_text(
                        translated_parser_output.json(indent=4, ensure_ascii=False)
                    )
                    logger.info(f"Saved translated output to {output_path}.")

                except cloudpathlib.exceptions.OverwriteNewerCloudError:
                    logger.info(
                        f"Tried to write to {output_path}, received OverwriteNewerCloudError, assuming a newer task "
                        f"definition is the one we want, continuing to process.")

    logger.info('Finished translation stage for ALL input tasks.')




