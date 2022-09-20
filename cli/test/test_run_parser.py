from pathlib import Path
import tempfile
from unittest import mock

import pytest
from click.testing import CliRunner

from cli.run_parser import main as cli_main
from src.base import ParserOutput

patcher = mock.patch(
    "src.translator.translate.translate_text",
    mock.MagicMock(return_value=["translated text"]),
)
patcher.start()


@pytest.mark.filterwarnings("ignore::urllib3.exceptions.InsecureRequestWarning")
def test_run_parser() -> None:
    """Test that the parsing CLI runs and outputs a file."""
    input_dir = str((Path(__file__).parent / "test_data" / "input").resolve())

    with tempfile.TemporaryDirectory() as output_dir:
        runner = CliRunner()
        result = runner.invoke(cli_main, [input_dir, output_dir, "--parallel"])

        assert result.exit_code == 0

        assert (Path(output_dir) / "test_html.json").exists()
        assert (Path(output_dir) / "test_pdf.json").exists()


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

        assert (Path(output_dir) / "test_html.json").exists()
        assert not (Path(output_dir) / "test_pdf.json").exists()


@pytest.mark.filterwarnings("ignore::urllib3.exceptions.InsecureRequestWarning")
def test_run_parser_skip_already_done(caplog) -> None:
    """Test that files which have already been parsed are skipped by default."""

    input_dir = str((Path(__file__).parent / "test_data" / "input").resolve())

    with tempfile.TemporaryDirectory() as output_dir:
        with open(Path(output_dir) / "test_pdf.json", "w") as f:
            f.write(
                ParserOutput.parse_obj(
                    {
                        "id": "test_pdf",
                        "url": "https://www.pdfs.org",
                        "document_name": "test_pdf",
                        "document_description": "test_pdf_description",
                        "content_type": "application/pdf",
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
                        "id": "test_html",
                        "url": "https://www.google.org",
                        "document_name": "test_html",
                        "document_description": "test_html_description",
                        "content_type": "text/html",
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
            "Found 2 documents that have already parsed. Skipping." in caplog.messages
        )
