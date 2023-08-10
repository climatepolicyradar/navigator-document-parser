from typing import List
import six
import logging

from cpr_data_access.parser_models import ParserOutput
from google.cloud import translate_v2


_LOGGER = logging.getLogger(__file__)


def translate_text(text: List[str], target_language: str) -> List[str]:
    """
    Translate text into the target language.

    Adapted from the Google Cloud docs: https://cloud.google.com/translate/docs/basic/translating-text#translating_text

    :param text: list of texts to translate. Recommended max length from Google is 5000 characters.
    :param target_language: target language. Must be an ISO 639-1 (2-letter) language code.
    :return: list of translated text
    """

    translate_client = translate_v2.Client()

    text = [
        _str.decode("utf-8") if isinstance(_str, six.binary_type) else _str
        for _str in text
    ]

    try:
        result = translate_client.translate(text, target_language=target_language)
        return [item["translatedText"] for item in result]
    except Exception as e:
        _LOGGER.exception(
            "Error translating text.",
            extra={
                "props": {
                    "text": text,
                    "target_language": target_language,
                    "error": str(e),
                }
            },
        )
        raise e


def translate_parser_output(
    parser_output: ParserOutput, target_language: str
) -> ParserOutput:
    """
    Translate a ParserOutput object into the target language.

    :param parser_output: ParserOutput object to translate
    :param target_language: target language. Must be an ISO 639-1 (2-letter) language code.
    :return: translated ParserOutput object
    """

    # A deep copy here prevents text blocks in the original ParserOutput object from being modified in place
    new_parser_output = parser_output.copy(deep=True)

    # Translate document name, document description and text
    new_parser_output.document_name = translate_text(
        [parser_output.document_name], target_language
    )[0]
    new_parser_output.document_description = translate_text(
        [parser_output.document_description], target_language
    )[0]

    if new_parser_output.html_data is not None:
        for block in new_parser_output.html_data.text_blocks:
            block.text = translate_text(block.text, target_language)
            block.language = target_language

    if new_parser_output.pdf_data is not None:
        for block in new_parser_output.pdf_data.text_blocks:
            block.text = translate_text(block.text, target_language)
            block.language = target_language

    # Set language and translation status of new ParserOutput object
    # TODO: is this language in the correct format?
    new_parser_output.languages = [target_language]
    new_parser_output.translated = True

    return new_parser_output
