import json
from typing import Union
import pytest

from azure.ai.formrecognizer import AnalyzeResult


def read_local_json_file(file_path: str) -> Union[list[dict], dict]:
    """Read a local json file and return the data."""
    with open(file_path) as json_file:
        data = json.load(json_file)
    return data


@pytest.fixture()
def one_page_analyse_result() -> AnalyzeResult:
    """Mock response for the analyse document from url endpoint."""
    data = read_local_json_file(
        "./cli/test/test_data/api_response/sample-one-page.json"
    )
    return AnalyzeResult.from_dict(data[0])
