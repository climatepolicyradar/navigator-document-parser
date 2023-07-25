import unittest
from unittest.mock import patch

from azure.ai.formrecognizer import AnalyzeResult

from src.base import PDFPage
from src.pdf_parser.azure_wrapper import call_api_with_error_handling, AzureApiWrapper


def test_call_api_with_error_handling_good_response(
    mock_azure_client: AzureApiWrapper,
    one_page_pdf_bytes: bytes,
    one_page_mock_analyse_result: AnalyzeResult
) -> None:
    """Test the retry logic and exception handling to the function when the api
    response is good."""

    response = call_api_with_error_handling(
        retries=3,
        func=mock_azure_client.analyze_document_from_url,
        doc_bytes=one_page_pdf_bytes,
        timeout=None
    )

    assert response == one_page_mock_analyse_result
    assert mock_azure_client.analyze_document_from_url.call_count is 1


def test_call_api_with_error_handling_bad_response(
    mock_azure_client: AzureApiWrapper,
    one_page_pdf_bytes: bytes,
) -> None:
    """Test the retry logic and exception handling to the function when the api
    response is bad."""
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


def test_analyze_document_from_url(
    mock_azure_client: AzureApiWrapper,
    one_page_mock_analyse_result: AnalyzeResult
) -> None:
    """Test that the document from url method returns the correct response."""
    response = mock_azure_client.analyze_document_from_url(
        "https://example.com/test.pdf"
    )

    assert mock_azure_client.analyze_document_from_url.call_count is 1
    assert response == one_page_mock_analyse_result


def test_analyze_document_from_bytes(
    mock_azure_client: AzureApiWrapper,
    one_page_mock_analyse_result: AnalyzeResult
) -> None:
    """Test that the document from bytes method returns the correct response."""
    response = mock_azure_client.analyze_document_from_bytes(
        bytes("Random Content".encode("UTF-8"))
    )

    assert response == one_page_mock_analyse_result


def test_document_split_one_page(
    mock_azure_client: AzureApiWrapper,
    one_page_mock_analyse_result: AnalyzeResult,
    mock_document_download_response_one_page: unittest.mock.Mock
) -> None:
    """Test that processing a document via url with the multi page function returns
    the correct response."""
    with patch('requests.get') as mock_get:
        mock_get.return_value = mock_document_download_response_one_page

        response = mock_azure_client.analyze_large_document_from_url_split(
            "https://example.com/test.pdf"
        )

        page_api_responses = response[0]
        merged_page_api_responses = response[1]

        assert isinstance(page_api_responses, list)
        assert len(page_api_responses) is 1
        assert isinstance(page_api_responses[0], PDFPage)
        assert page_api_responses[0].page_number is 1
        assert page_api_responses[0].extracted_content is one_page_mock_analyse_result

        assert isinstance(merged_page_api_responses, AnalyzeResult)


def test_document_split_two_page(
    mock_azure_client: AzureApiWrapper,
    one_page_mock_analyse_result: AnalyzeResult,
    mock_document_download_response_two_page: unittest.mock.Mock
) -> None:
    """Test that processing a document via url with the split page functionality
    returns the correct response.

    We mock the response from the document download request as well as the response
    from the azure api to extract content from the page.
    """
    with patch('requests.get') as mock_get:
        mock_get.return_value = mock_document_download_response_two_page

        response = mock_azure_client.analyze_large_document_from_url_split(
            "https://example.com/test.pdf"
        )

        page_api_responses = response[0]
        merged_page_api_responses = response[1]

        assert isinstance(page_api_responses, list)
        assert len(page_api_responses) is 2
        for i, page_response in enumerate(page_api_responses):
            assert isinstance(page_response, PDFPage)
            assert page_response.page_number is i + 1
            assert page_response.extracted_content is one_page_mock_analyse_result

        assert isinstance(merged_page_api_responses, AnalyzeResult)
