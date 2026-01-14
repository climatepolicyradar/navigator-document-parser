from pathlib import Path
from typing import List
from unittest import mock

import pytest
from cpr_sdk.parser_models import ParserOutput
from google.cloud import translate_v2
from google.oauth2.service_account import Credentials

from src.translator.translate import should_translate_text, translate_parser_output
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend


@pytest.fixture
def google_translate_credentials_mock() -> Credentials:
    """Create an instance of google translate credentials from mocked values"""

    key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend(),
    )

    pem_string = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")

    credentials_info = {
        "type": "service_account",
        "project_id": "dummy-project",
        "private_key_id": "dummy-private-key-id",
        "private_key": pem_string,
        "client_email": "dummy-service-account@dummy-project.iam.gserviceaccount.com",
        "client_id": "123456789012345678901",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": (
            "https://www.googleapis.com/robot/v1/metadata/x509/"
            "dummy-service-account%40dummy-project.iam.gserviceaccount.com"
        ),
    }

    return Credentials.from_service_account_info(credentials_info)


def fake_translate_text(
    client: translate_v2.Client, text: List[str], target_language: str
) -> List[str]:
    """Mock translate_text function."""
    return [f"translated to {target_language}: {t}" for t in text]


def test_translate_parser_output(google_translate_credentials_mock: Credentials) -> None:
    """Test that translate_parser_output translates the text, document name and document description."""

    # Run translation with mocked Google Cloud Translate
    with mock.patch(
        "src.translator.translate.translate_text",
        wraps=fake_translate_text,
    ):
        test_file_path = (
            Path(__file__).parent.parent.parent.parent
            / "cli"
            / "test"
            / "test_data"
            / "output"
            / "test_html.json"
        )

        parser_output = ParserOutput.model_validate_json(test_file_path.read_text())

        translated_parser_output = translate_parser_output(
            parser_output,
            "fr",
            google_translate_credentials_mock,
        )

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
                for original, translated in zip(
                    original_text, translated_text, strict=False
                )
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
        "document_source_url",
        "document_cdn_object",
        "document_md5_sum",
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
            translated_parser_output.html_data.text_blocks,
            strict=False,  # type: ignore
        ):
            assert getattr(text_block, text_block_attr) == getattr(
                translated_text_block, text_block_attr
            )


@pytest.mark.parametrize(
    "text, expected",
    [
        ("-", False),
        ("6", False),
        (".", False),
        ("12.1123", False),
        ("$", False),
        ("!!!", False),
        ("123456", False),
        ("3.14159", False),
        ("hello", True),
        ("world!", True),
        ("hello world", True),
        ("text", True),
        ("bonjour", True),
        ("hello, world!", True),
        ("hello, world! 123", True),
        ("12312!", False),
        ("(12)", False),
    ],
)
def test_should_translate_text(text: str, expected: bool) -> None:
    """Test should_translate_text function with various inputs."""
    assert should_translate_text(text) == expected
