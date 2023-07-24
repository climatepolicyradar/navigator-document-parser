import io
import sys
from io import BytesIO
import time
from typing import Tuple, Sequence, Union, Optional
import logging
import requests as requests

from azure.ai.formrecognizer import AnalyzeResult, DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential

from src.pdf_parser.utils import split_into_pages, PDFPage, merge_responses

logger = logging.getLogger(__name__)


def poller_loop(
    poller: Union[
        DocumentAnalysisClient.begin_analyze_document_from_url,
        DocumentAnalysisClient.begin_analyze_document
    ]
) -> None:
    """Poll the status of the poller until it is done."""
    counter = 0
    logger.info(f'Poller status {poller.status()}...')
    while not poller.done():
        time.sleep(0.2)
        counter += 1
        if counter % 50 == 0:
            logger.info(f'Poller status {poller.status()}...')
    logger.info(f'Poller status {poller.status()}...')


def call_api_with_error_handling(retries: int, func, *args, **kwargs) -> None:
    """Call an API function with retries and error handling."""
    logger.info(f'Calling API function with retries...', extra={"props": {"retries": retries}})
    for i in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f'Error occurred while calling API function...', extra={"props": {"error": str(e)}})
            if i == retries - 1:
                raise e


class AzureApiWrapper:
    """Wrapper for Azure Form Extraction API."""

    def __init__(self, key: str, endpoint: str):
        logger.info(f'Initializing Azure API wrapper with endpoint...', extra={"props": {"endpoint": endpoint}})
        self.document_analysis_client = DocumentAnalysisClient(
            endpoint=endpoint, credential=AzureKeyCredential(key),
        )

    def analyze_document_from_url(self, doc_url: str, timeout: Optional[Union[int, None]] = None) -> AnalyzeResult:
        """Analyze a pdf document accessible by an endpoint."""
        logger.info('Analyzing document from url...', extra={"props": {"url": doc_url}})
        poller = self.document_analysis_client.begin_analyze_document_from_url(
            'prebuilt-document', doc_url,
        )

        poller_loop(poller)

        return poller.result(timeout=timeout)

    def analyze_document_from_bytes(self, doc_bytes: bytes, timeout: Optional[Union[int, None]] = None) -> AnalyzeResult:
        """Analyze a pdf document in the form of bytes."""
        logger.info('Analyzing document from bytes...', extra={"props": {"bytes_size": sys.getsizeof(doc_bytes)}})
        poller = self.document_analysis_client.begin_analyze_document(
            'prebuilt-document', doc_bytes,
        )

        poller_loop(poller)

        return poller.result(timeout=timeout)

    def analyze_large_document_from_url_split(
            self, doc_url: str, timeout: Optional[Union[int, None]] = None) -> Tuple[Sequence[PDFPage], AnalyzeResult]:
        """Analyze a large pdf document accessible by an endpoint by splitting into individual pages."""
        logger.info(
            'Analyzing large document from url by splitting into individual pages...',
            extra={"props": {"url": doc_url}}
        )
        try:
            resp = requests.get(doc_url)
            pdf_bytes = BytesIO(resp.content)

            pages_dict = split_into_pages(document_bytes=pdf_bytes)
            logger.info(f'Number of pages: {len(pages_dict)}')
            page_api_responses = [
                PDFPage(
                    page_number=page_num,
                    extracted_content=call_api_with_error_handling(
                        func=self.analyze_document_from_bytes,
                        retries=3,
                        doc_bytes=page_bytes,
                        timeout=timeout,
                    ),
                )
                for page_num, page_bytes in pages_dict.items()
            ]

            return page_api_responses, merge_responses(page_api_responses)

        except Exception as e:
            logger.error(
                'Error occurred while analyzing large document from url by splitting into individual pages...',
                extra={"props": {"url": doc_url}}
            )
            raise e

    def analyze_large_document_from_bytes_split(
            self, doc_bytes: bytes, timeout: Optional[Union[int, None]] = None) -> Tuple[Sequence[PDFPage], AnalyzeResult]:
        """Analyze a large pdf document in the bytes form by splitting into individual pages."""
        logger.info(
            'Analyzing large document from bytes by splitting into individual pages...',
            extra={"props": {"bytes_size": sys.getsizeof(doc_bytes)}}
        )
        try:
            pages_dict = split_into_pages(document_bytes=io.BytesIO(doc_bytes))
            logger.info(f'Number of pages: {len(pages_dict)}')
            page_api_responses = [
                PDFPage(
                    page_number=page_num,
                    extracted_content=call_api_with_error_handling(
                        func=self.analyze_document_from_bytes,
                        retries=3,
                        doc_bytes=page_bytes,
                        timeout=timeout,
                    ),
                )
                for page_num, page_bytes in pages_dict.items()
            ]

            return page_api_responses, merge_responses(page_api_responses)

        except Exception as e:
            logger.error(
                'Error occurred while analyzing large document from bytes by splitting into individual pages...',
                extra={"props": {"bytes_size": sys.getsizeof(doc_bytes)}}
            )
            raise e
