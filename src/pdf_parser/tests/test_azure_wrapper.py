def test_analyze_document_from_url(mock_azure_client, mock_response_analyse_document_from_url):
    response = mock_azure_client.analyze_document_from_url("https://example.com/test.pdf")

    assert mock_azure_client.analyze_document_from_url.call_count is 1
    assert response == mock_response_analyse_document_from_url
