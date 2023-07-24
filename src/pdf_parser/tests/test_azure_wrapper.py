import unittest
from mock import call, patch

from src.pdf_parser.azure_wrapper import AzureApiWrapper


class AzureApiWrapperApiTest(unittest.TestCase):

    @patch('src.pdf_parser.azure_wrapper.AzureApiWrapper.analyze_document_from_url')
    def test_analyze_document_from_url(self, mocked_api_func):

        response = AzureApiWrapper('user', 'pass').analyze_document_from_url("https://example.com/test.pdf")

        self.assertTrue(mocked_api_func.called)
        self.assertEqual(
            mocked_api_func.call_args_list,
            [call('user', 'pass')]
        )
        self.assertEqual(mocked_api_func.return_value, response)
