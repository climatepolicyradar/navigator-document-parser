from pathlib import Path
import tempfile

import pytest
from click.testing import CliRunner

from cli.run_parser import main as cli_main


@pytest.mark.filterwarnings("ignore::urllib3.exceptions.InsecureRequestWarning")
def test_run_parser():
    """Test that the parsing CLI runs and outputs a file."""
    input_dir = str((Path(__file__).parent / "data" / "input").resolve())

    with tempfile.TemporaryDirectory() as output_dir:
        runner = CliRunner()
        result = runner.invoke(cli_main, [input_dir, output_dir])

        assert result.exit_code == 0

        assert (Path(output_dir) / "test_id.json").exists()
