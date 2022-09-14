from pathlib import Path
import tempfile

import pytest
from click.testing import CliRunner

from cli.run_parser import main as cli_main
from src.base import HTMLParserOutput, PDFParserOutput


@pytest.mark.filterwarnings("ignore::urllib3.exceptions.InsecureRequestWarning")
def test_run_parser() -> None:
    """Test that the parsing CLI runs and outputs a file."""
    input_dir = str((Path(__file__).parent / "test_data" / "input").resolve())

    with tempfile.TemporaryDirectory() as output_dir:
        runner = CliRunner()
        result = runner.invoke(cli_main, [input_dir, output_dir])

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
                PDFParserOutput.parse_obj(
                    {
                        "id": "test_pdf",
                        "url": "https://www.pdfs.org",
                        "languages": ["en"],
                        "text_blocks": [],
                        "document_slug": "slug",
                        "page_metadata": [],
                        "md5sum": "abcdefg",
                    }
                ).json()
            )

        with open(Path(output_dir) / "test_html.json", "w") as f:
            f.write(
                HTMLParserOutput.parse_obj(
                    {
                        "id": "test_html",
                        "url": "https://www.google.org",
                        "languages": ["en"],
                        "text_blocks": [],
                        "date": None,
                        "has_valid_text": False,
                        "document_slug": "slug",
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
