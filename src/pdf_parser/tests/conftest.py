from unittest.mock import MagicMock

import pytest
import json

from azure.ai.formrecognizer import AnalyzeResult

from src.pdf_parser.azure_wrapper import AzureApiWrapper


def read_local_json_file(file_path: str) -> list[dict[dict]]:
    """Read a local json file and return the data."""
    with open(file_path) as json_file:
        data = json.load(json_file)
    return data


@pytest.fixture()
def mock_response_analyse_document_from_url() -> AnalyzeResult:
    """Mock response for the analyse document from url endpoint."""
    data = read_local_json_file("./data/sample-layout.json")
    return AnalyzeResult.from_dict(data[0])


@pytest.fixture()
def mock_azure_client(mock_response_analyse_document_from_url) -> AzureApiWrapper:
    azure_client = AzureApiWrapper('user', 'pass')
    azure_client.analyze_document_from_url = MagicMock(return_value=mock_response_analyse_document_from_url)
    return azure_client
