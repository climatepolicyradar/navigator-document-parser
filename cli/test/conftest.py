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


@pytest.fixture()
def backend_document_json() -> dict:
    """BackendDocument JSON for testing."""
    return {
        "publication_ts": "2013-01-01T00:00:00",
        "name": "test_pdf",
        "description": "test_pdf_description",
        "source_url": "https://www.pdfs.org",
        "download_url": None,
        "url": None,
        "md5_sum": None,
        "type": "EU Decision",
        "source": "CCLW",
        "import_id": "test_pdf",
        "family_import_id": "TESTCCLW.family.4.0",
        "category": "Law",
        "geography": "EUR",
        "languages": [
            "English"
        ],
        "metadata": {
            "hazards": [],
            "frameworks": [],
            "instruments": [
                "Capacity building|Governance"
            ],
            "keywords": [
                "Adaptation"
            ],
            "sectors": [
                "Economy-wide"
            ],
            "topics": [
                "Adaptation"
            ]
        },
        "slug": "dummy_slug"
    }
