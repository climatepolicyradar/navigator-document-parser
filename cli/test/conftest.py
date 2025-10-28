import json
from pathlib import Path
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
    """Backend Document JSON for testing."""
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
        "family_slug": "slug_TESTCCLW.family.4.0",
        "category": "Law",
        "geography": "EUR",
        "languages": ["English"],
        "metadata": {
            "hazards": [],
            "frameworks": [],
            "instruments": ["Capacity building|Governance"],
            "keywords": ["Adaptation"],
            "sectors": ["Economy-wide"],
            "topics": ["Adaptation"],
        },
        "slug": "dummy_slug",
    }


@pytest.fixture()
def test_input_dir() -> Path:
    return (Path(__file__).parent / "test_data" / "input").resolve()


@pytest.fixture()
def azure_api_cache_dir() -> str:
    """The directory where the azure api response cache is stored."""
    return "azure_api_response_cache"


@pytest.fixture()
def test_azure_api_response_dir(azure_api_cache_dir) -> Path:
    return (Path(__file__).parent / "test_data" / azure_api_cache_dir).resolve()


@pytest.fixture()
def archived_file_name_pattern() -> str:
    return r"^\d+\.json$"


@pytest.fixture()
def expected_pipeline_metadata_keys() -> set[str]:
    """The names of the pipeline metadata keys that are expected in the output."""
    return {"azure_model_id", "parsing_date", "azure_api_version"}
