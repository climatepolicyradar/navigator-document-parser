import logging
from pathlib import Path
from typing import Set, Sequence, Union

import cloudpathlib.exceptions
from cloudpathlib import CloudPath
from tqdm.auto import tqdm

from src.base import ParserOutput
from src.config import TARGET_LANGUAGES, LOGGING_LEVEL
from src.translator.translate import translate_parser_output

_LOGGER = logging.getLogger(__name__)
level = logging.getLevelName(LOGGING_LEVEL)
_LOGGER.setLevel(level)


def should_be_translated(document: ParserOutput) -> bool:
    """
    Determine if a document should be translated.

    If the document has not already been translated and has not null source url, then it should be translated.
    """
    if document.translated or document.document_source_url is None:
        return False
    return True


def identify_translation_languages(
    document: ParserOutput, target_languages: Set
) -> Set:
    """Determine the languages to translate a document to.

    We subtract the current document languages from the target languages to determine the languages to translate to.
    E.g. doc.languages=['fr'] and target_languages=['en'] -> ['en'] - ['fr'] -> ['en'] (translate to English)
    E.g. doc.languages=['en'] and target_languages=['en'] -> ['en'] - ['en'] -> [] (no languages to translate to)

    If there are no detected document languages then we translate to all target languages.
    """
    # TODO: how do we deal with the fact that parser outputs can contain multiple languages here?
    if document.languages and len(document.languages) == 1:
        minus = set(document.languages)
        _LOGGER.debug(
            f"Removing {minus if minus is not None else None} from {target_languages}."
        )
        target_languages = target_languages - set(document.languages)
    return target_languages


def translate_parser_outputs(
    task_output_paths: Sequence[Union[Path, CloudPath]], redo: bool = False
) -> None:
    """
    Translate parser outputs saved in the output directory, and save the translated outputs to the output directory.

    :param task_output_paths: A list of the paths to the parser outputs for this current instance to translate.
    """
    _target_languages = set(TARGET_LANGUAGES)

    for path in tqdm(task_output_paths):
        _LOGGER.debug(f"Translator processing - {path}.")

        try:
            parser_output = ParserOutput.parse_raw(path.read_text())
            _LOGGER.debug(
                f"Successfully parsed {path} from output dir during translation processing."
            )
        except FileNotFoundError:
            _LOGGER.error(
                f"Could not find {path} in output dir during translation processing."
            )
            continue

        if should_be_translated(parser_output):
            _LOGGER.info(f"Document should be translated: {path}")

            target_languages = identify_translation_languages(
                parser_output, _target_languages
            )
            _LOGGER.debug(f"Target languages: {target_languages} for {path}")

            _translate_to_target_languages(
                path, parser_output, target_languages, redo=redo
            )

    _LOGGER.info("Finished translation stage for ALL input tasks.")


def _translate_to_target_languages(
    path: Union[Path, CloudPath],
    parser_output: ParserOutput,
    target_languages: set[str],
    redo: bool = False,
) -> None:
    for target_language in target_languages:
        try:
            _LOGGER.info(f"Translating {path} to {target_language}.")

            output_path = path.with_name(
                f"{path.stem}_translated_{target_language}.json"
            )
            if output_path.exists() and not redo:  # type: ignore
                _LOGGER.info(
                    f"Skipping translating {output_path} because it already exists."
                )
                continue

            translated_parser_output = translate_parser_output(
                parser_output, target_language
            )
            _LOGGER.info(f"Translated {path} to {target_language}.")

            try:
                output_path.write_text(  # type: ignore
                    translated_parser_output.json(indent=4, ensure_ascii=False)
                )
                _LOGGER.info(f"Saved translated output to {output_path}.")

            except cloudpathlib.exceptions.OverwriteNewerCloudError:
                _LOGGER.info(
                    f"Tried to write to {output_path}, received OverwriteNewerCloudError, "
                    "assuming a newer task definition is the one we want, continuing to "
                    "process."
                )
        except Exception:
            _LOGGER.exception(
                "Failed to successfully translate due to a raised exception",
                extra={
                    "props": {
                        "input_path": f"{path}",
                        "target_language": target_language,
                    }
                },
            )
