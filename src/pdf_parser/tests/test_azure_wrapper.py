import unittest
from unittest.mock import MagicMock, patch

from azure.ai.formrecognizer import AnalyzeResult

from src.base import PDFPage
from src.pdf_parser.azure_wrapper import call_api_with_error_handling

# TODO Test that call api with errors retries the set amount of times
#   Test that the poller continues until done


def test_call_api_with_error_handling_good_response(mock_azure_client, one_page_pdf_bytes):
    """Test the retry logic and exception handling to the function when the api response is good."""

    call_api_with_error_handling(
        retries=3,
        func=mock_azure_client.analyze_document_from_url,
        doc_bytes=one_page_pdf_bytes,
        timeout=None
    )

    assert mock_azure_client.analyze_document_from_url.call_count is 1


def test_call_api_with_error_handling_bad_response(mock_azure_client, one_page_pdf_bytes):
    """Test the retry logic and exception handling to the function when the api response is bad."""
    retries = 3
    exception_to_raise = Exception("Simulated API error")

    mock_azure_client.analyze_document_from_url.side_effect = exception_to_raise

    exception_raised = None
    try:
        call_api_with_error_handling(
            retries=retries,
            func=mock_azure_client.analyze_document_from_url,
            doc_bytes=one_page_pdf_bytes,
            timeout=None
        )
    except Exception as e:
        exception_raised = e

    assert exception_raised == exception_to_raise
    assert mock_azure_client.analyze_document_from_url.call_count is retries
    assert mock_azure_client.analyze_document_from_url


def test_analyze_document_from_url(mock_azure_client, one_page_mock_analyse_result):
    """Test that the document from url method returns the correct response."""
    response = mock_azure_client.analyze_document_from_url("https://example.com/test.pdf")

    assert mock_azure_client.analyze_document_from_url.call_count is 1
    assert response == one_page_mock_analyse_result


def test_analyze_document_from_bytes(mock_azure_client, one_page_mock_analyse_result):
    """Test that the document from bytes method returns the correct response."""
    response = mock_azure_client.analyze_document_from_bytes(bytes("Random Content".encode("UTF-8")))

    assert response == one_page_mock_analyse_result


def test_document_split_one_page(mock_azure_client, one_page_mock_analyse_result, one_page_pdf_bytes):
    """Test that processing a document via url with the multi page function returns the correct response."""
    # TODO can this go in a pytest fixture
    with patch('requests.get') as mock_get:
        # Create a mock Response object
        mock_response = unittest.mock.Mock()
        mock_response.content = one_page_pdf_bytes

        # Set the status code and other attributes as needed for your test
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'application/pdf'}

        # Configure the mock get method to return the mock Response
        mock_get.return_value = mock_response

        page_api_responses, merged_page_api_responses = mock_azure_client.analyze_large_document_from_url_split(
            "https://example.com/test.pdf"
        )

        assert isinstance(page_api_responses, list)
        assert len(page_api_responses) is 1
        assert isinstance(page_api_responses[0], PDFPage)
        assert page_api_responses[0].page_number is 1
        assert page_api_responses[0].extracted_content is one_page_mock_analyse_result
        assert isinstance(merged_page_api_responses, AnalyzeResult)

