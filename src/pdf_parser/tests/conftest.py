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


def read_pdf_to_bytes(file_path: str) -> bytes:
    """Read a pdf to bytes from a local path."""
    with open(file_path, 'rb') as file:
        pdf_bytes = file.read()
    return pdf_bytes


@pytest.fixture()
def one_page_pdf_bytes() -> bytes:
    """Content for the sample one page pdf"""
    return read_pdf_to_bytes("data/sample-one-page.pdf")


@pytest.fixture()
def one_page_mock_analyse_result() -> AnalyzeResult:
    """Mock response for the analyse document from url endpoint."""
    data = read_local_json_file("data/sample-one-page.json")
    return AnalyzeResult.from_dict(data[0])


@pytest.fixture()
def mock_azure_client(one_page_mock_analyse_result) -> AzureApiWrapper:
    """A mock client to the azure form recognizer api with mocked responses from the api endpoints."""
    azure_client = AzureApiWrapper('user', 'pass')
    azure_client.analyze_document_from_url = MagicMock(return_value=one_page_mock_analyse_result)
    azure_client.analyze_document_from_bytes = MagicMock(return_value=one_page_mock_analyse_result)
    return azure_client
