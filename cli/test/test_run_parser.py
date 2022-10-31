from pathlib import Path
import tempfile
from unittest import mock
from typing import Sequence, Union
import pytest
from click.testing import CliRunner

from cloudpathlib.local import LocalS3Path

from cli.run_parser import main as cli_main
from cli.translate_outputs import should_be_translated, identify_translation_languages

from src.base import ParserOutput, HTMLData
from src.config import TARGET_LANGUAGES

patcher = mock.patch(
    "src.translator.translate.translate_text",
    mock.MagicMock(return_value=["translated text"]),
)
patcher.start()


@pytest.fixture()
def test_input_dir() -> Path:
    return (Path(__file__).parent / "test_data" / "input").resolve()


@pytest.mark.filterwarnings("ignore::urllib3.exceptions.InsecureRequestWarning")
def test_run_parser_local(test_input_dir) -> None:
    """Test that the parsing CLI runs and outputs a file."""
    with tempfile.TemporaryDirectory() as output_dir:
        runner = CliRunner()

        result = runner.invoke(
            cli_main, [str(test_input_dir), output_dir, "--parallel"]
        )

        assert result.exit_code == 0

        # Default config is to translate to English, and the HTML doc is already in English - so we just expect a translation of the PDF
        assert set(Path(output_dir).glob("*.json")) == {
            Path(output_dir) / "test_html.json",
            Path(output_dir) / "test_pdf.json",
            Path(output_dir) / "test_no_content_type.json",
            Path(output_dir) / "test_pdf_translated_en.json",
        }

        for output_file in Path(output_dir).glob("*.json"):
            assert ParserOutput.parse_file(output_file)


@pytest.mark.filterwarnings("ignore::urllib3.exceptions.InsecureRequestWarning")
def test_run_parser_s3(test_input_dir) -> None:
    """Test that the parsing CLI runs and outputs a file."""

    input_dir = "s3://test-bucket/test-input-dir"
    output_dir = "s3://test-bucket/test-output-dir"

    # Copy test data to mock of S3 path
    input_file_path = LocalS3Path(f"{input_dir}/test_html.json")
    input_file_path.write_text((test_input_dir / "test_html.json").read_text())

    with mock.patch("cli.run_parser.S3Path", LocalS3Path):
        runner = CliRunner()
        result = runner.invoke(cli_main, [input_dir, output_dir, "--s3", "--parallel"])
        assert result.exit_code == 0
        assert set(LocalS3Path(output_dir).glob("*.json")) == {
            LocalS3Path(output_dir) / "test_html.json"
        }


@pytest.mark.filterwarnings("ignore::urllib3.exceptions.InsecureRequestWarning")
def test_run_parser_specific_files() -> None:
    """Test that using the `--files` flag only parses the files that have been specified."""

    input_dir = str((Path(__file__).parent / "test_data" / "input").resolve())

    with tempfile.TemporaryDirectory() as output_dir:
        runner = CliRunner()
        result = runner.invoke(
            cli_main, [input_dir, output_dir, "--parallel", "--files", "test_html.json"]
        )

        assert result.exit_code == 0

        assert set(Path(output_dir).glob("*.json")) == {
            Path(output_dir) / "test_html.json"
        }


@pytest.mark.filterwarnings("ignore::urllib3.exceptions.InsecureRequestWarning")
def test_run_parser_skip_already_done(caplog) -> None:
    """Test that files which have already been parsed are skipped by default."""

    input_dir = str((Path(__file__).parent / "test_data" / "input").resolve())

    with tempfile.TemporaryDirectory() as output_dir:
        with open(Path(output_dir) / "test_pdf.json", "w") as f:
            f.write(
                ParserOutput.parse_obj(
                    {
                        "document_id": "test_pdf",
                        "document_metadata": {},
                        "document_source_url": "https://www.pdfs.org",
                        "document_cdn_object": "test_pdf.pdf",
                        "document_md5_sum": "abcdefghijk",
                        "document_name": "test_pdf",
                        "document_description": "test_pdf_description",
                        "document_content_type": "application/pdf",
                        "languages": ["en"],
                        "document_slug": "slug",
                        "pdf_data": {
                            "text_blocks": [],
                            "page_metadata": [],
                            "md5sum": "abcdefg",
                        },
                    }
                ).json()
            )

        with open(Path(output_dir) / "test_html.json", "w") as f:
            f.write(
                ParserOutput.parse_obj(
                    {
                        "document_id": "test_html",
                        "document_metadata": {},
                        "document_source_url": "https://www.google.org",
                        "document_cdn_object": None,
                        "document_md5_sum": None,
                        "document_name": "test_html",
                        "document_description": "test_html_description",
                        "document_content_type": "text/html",
                        "languages": ["en"],
                        "document_slug": "slug",
                        "html_data": {
                            "text_blocks": [],
                            "detected_title": "",
                            "detected_date": None,
                            "has_valid_text": False,
                        },
                    }
                ).json()
            )

        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                input_dir,
                output_dir,
                "--parallel",
            ],
        )

        assert result.exit_code == 0

        assert (
                "Skipping 2 documents that have already been parsed." in caplog.messages
        )


_target_languages = set(TARGET_LANGUAGES)


def get_parser_output(translated: bool, source_url: Union[str, None], languages: Sequence[str]) -> ParserOutput:
    """Generate the parser output objects for the tests given input variables."""
    return ParserOutput(
        document_id='sdf',
        document_metadata={},
        document_name='sdf',
        document_description='sdf',
        document_source_url=source_url,
        document_cdn_object='sdf',
        document_content_type="text/html",
        document_md5_sum='sdf',
        document_slug='sdf',
        languages=languages,
        translated=translated,
        html_data=HTMLData(
            text_blocks=[],
            detected_date=None,
            detected_title="",
            has_valid_text=False,
        ),
        pdf_data=None
    )


@pytest.mark.filterwarnings("ignore::urllib3.exceptions.InsecureRequestWarning")
def test_should_be_translated() -> None:
    """Tests that the output from the function is correct for known input documents."""
    doc_1 = get_parser_output(translated=False, source_url="https://www.google.org", languages=['fr'])
    assert should_be_translated(doc_1) is True

    doc_2 = get_parser_output(translated=False, source_url=None, languages=['fr'])
    assert should_be_translated(doc_2) is False

    doc_3 = get_parser_output(translated=False, source_url="https://www.google.org", languages=['English'])
    assert should_be_translated(doc_3) is True

    doc_4 = get_parser_output(translated=True, source_url="https://www.google.org", languages=['fr'])
    assert should_be_translated(doc_4) is False


@pytest.mark.filterwarnings("ignore::urllib3.exceptions.InsecureRequestWarning")
def test_identify_target_languages() -> None:
    """Tests that the output from the function is correct for known input documents."""
    doc_1 = get_parser_output(translated=False, source_url="https://www.google.org", languages=['fr'])
    assert identify_translation_languages(doc_1, _target_languages) == {'en'}

    doc_2 = get_parser_output(translated=False, source_url="https://www.google.org", languages=['en'])
    assert identify_translation_languages(doc_2, _target_languages) == set()
