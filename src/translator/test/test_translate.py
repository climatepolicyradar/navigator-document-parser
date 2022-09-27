from typing import List
from pathlib import Path
from unittest import mock

from src.base import ParserOutput
from src.translator.translate import translate_parser_output


def fake_translate_text(text: List[str], target_language: str) -> List[str]:
    """Mock translate_text function."""
    return [f"translated to {target_language}: {t}" for t in text]


def test_translate_parser_output() -> None:
    """Test that translate_parser_output translates the text, document name and document description."""

    # Run translation with mocked Google Cloud Translate
    with mock.patch(
        "src.translator.translate.translate_text",
        wraps=fake_translate_text,
    ):
        parser_output = ParserOutput.parse_file(
            Path(__file__).parent.parent.parent.parent
            / "cli"
            / "test"
            / "test_data"
            / "output"
            / "test_html.json"
        )

        translated_parser_output = translate_parser_output(parser_output, "fr")

    # Check attributes that should be translated
    assert (
        translated_parser_output.document_name
        == "translated to fr: " + parser_output.document_name
    )
    assert (
        translated_parser_output.document_description
        == "translated to fr: " + parser_output.document_description
    )

    for idx in range(len(parser_output.html_data.text_blocks)):  # type: ignore
        original_text = parser_output.html_data.text_blocks[idx].text  # type: ignore
        translated_text = translated_parser_output.html_data.text_blocks[idx].text  # type: ignore

        assert all(
            [
                translated == "translated to fr: " + original
                for original, translated in zip(original_text, translated_text)
            ]
        )

    # Check attributes that should have changed
    assert translated_parser_output.languages == ["fr"]
    assert translated_parser_output.translated is True
    assert all(
        [
            text_block.language == "fr"
            for text_block in translated_parser_output.html_data.text_blocks  # type: ignore
        ]
    )

    # Check attributes that should not have changed
    for attr in (
        "document_id",
        "document_url",
        "document_slug",
        "document_content_type",
    ):
        assert getattr(translated_parser_output, attr) == getattr(parser_output, attr)

    for html_attr in ("detected_title", "detected_date", "has_valid_text"):
        assert getattr(translated_parser_output.html_data, html_attr) == getattr(
            parser_output.html_data, html_attr
        )

    for text_block_attr in ("text_block_id", "type", "type_confidence"):
        for text_block, translated_text_block in zip(
            parser_output.html_data.text_blocks,  # type: ignore
            translated_parser_output.html_data.text_blocks,  # type: ignore
        ):
            assert getattr(text_block, text_block_attr) == getattr(
                translated_text_block, text_block_attr
            )
