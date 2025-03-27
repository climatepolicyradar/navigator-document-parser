import logging
import string
from typing import List

import six
from cpr_sdk.parser_models import ParserOutput
from google.cloud import translate_v2
from tenacity import retry, stop_after_attempt, wait_random_exponential

_LOGGER = logging.getLogger(__file__)


@retry(
    stop=stop_after_attempt(4),
    wait=wait_random_exponential(multiplier=1, min=1, max=10),
)
def translate_text(
    translate_client: translate_v2.Client, text: List[str], target_language: str
) -> List[str]:
    """
    Translate text into the target language.

    Adapted from the Google Cloud docs: https://cloud.google.com/translate/docs/basic/translating-text#translating_text

    :param text: list of texts to translate. Recommended max length from Google is 5000 characters.
    :param target_language: target language. Must be an ISO 639-1 (2-letter) language code.
    :return: list of translated text
    """

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


def should_translate_text(text: str) -> bool:
    """
    Identify whether we should translate text.

    For example punctuation and numbers shouldn't be translated as they are the same in
    most languages.
    """
    if all(char in string.punctuation for char in text):
        return False

    try:
        float(text)
        return False
    except ValueError:
        pass

    return True


def translate_parser_output(
    parser_output: ParserOutput, target_language: str
) -> ParserOutput:
    """
    Translate a ParserOutput object into the target language.

    :param parser_output: ParserOutput object to translate
    :param target_language: target language. Must be an ISO 639-1 (2-letter) language code.
    :return: translated ParserOutput object
    """
    translate_client = translate_v2.Client()

    # A deep copy here prevents text blocks in the original ParserOutput object from being modified in place
    new_parser_output = parser_output.model_copy(deep=True)

    # Translate document name, document description and text
    new_parser_output.document_name = translate_text(
        translate_client, [parser_output.document_name], target_language
    )[0]
    new_parser_output.document_description = translate_text(
        translate_client, [parser_output.document_description], target_language
    )[0]

    if new_parser_output.html_data is not None:
        for block in new_parser_output.html_data.text_blocks:
            if all([should_translate_text(text) for text in block.text]):
                block.text = translate_text(
                    translate_client, block.text, target_language
                )
            block.language = target_language

    if new_parser_output.pdf_data is not None:
        for block in new_parser_output.pdf_data.text_blocks:
            if all([should_translate_text(text) for text in block.text]):
                block.text = translate_text(
                    translate_client, block.text, target_language
                )
            block.language = target_language

    # Set language and translation status of new ParserOutput object
    # TODO: is this language in the correct format?
    new_parser_output.languages = [target_language]
    new_parser_output.translated = True

    return new_parser_output
