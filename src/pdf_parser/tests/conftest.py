import unittest
from typing import Union
from unittest.mock import MagicMock

import pytest
import json

from azure.ai.formrecognizer import AnalyzeResult, DocumentParagraph, DocumentTable

from src.pdf_parser.azure_wrapper import AzureApiWrapper


def read_local_json_file(file_path: str) -> Union[list[dict[dict]], dict]:
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
def two_page_pdf_bytes() -> bytes:
    """Content for the sample two page pdf"""
    return read_pdf_to_bytes("data/sample-two-page.pdf")


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


@pytest.fixture
def mock_document_download_response_one_page(one_page_pdf_bytes) -> unittest.mock.Mock:
    """Create a mock response to a download request for a pdf document with one page."""
    # Create a mock Response object
    mock_response = unittest.mock.Mock()
    mock_response.content = one_page_pdf_bytes

    # Set the status code and other attributes as needed for your test
    mock_response.status_code = 200
    mock_response.headers = {'content-type': 'application/pdf'}

    return mock_response


@pytest.fixture
def mock_document_download_response_two_page(two_page_pdf_bytes) -> unittest.mock.Mock:
    """Create a mock response to a download request for a pdf document with two page."""
    # Create a mock Response object
    mock_response = unittest.mock.Mock()
    mock_response.content = two_page_pdf_bytes

    # Set the status code and other attributes as needed for your test
    mock_response.status_code = 200
    mock_response.headers = {'content-type': 'application/pdf'}

    return mock_response


@pytest.fixture
def document_paragraph() -> DocumentParagraph:
    """Construct a document paragraph object."""
    data = read_local_json_file("data/document-paragraph.json")
    return DocumentParagraph.from_dict(data)


@pytest.fixture
def document_table() -> DocumentTable:
    """Construct a document table object."""
    data = read_local_json_file("data/document-table.json")
    return DocumentTable.from_dict(data)
