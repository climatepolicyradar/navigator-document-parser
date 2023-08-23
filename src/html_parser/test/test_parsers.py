import pytest
from cpr_data_access.parser_models import ParserInput, ParserOutput

from src.base import HTMLParser
from src.html_parser.newsplease import NewsPleaseParser
from src.html_parser.readability import ReadabilityParser
from src.html_parser.combined import CombinedParser


@pytest.mark.parametrize(
    "url",
    [
        # "https://www.legislation.gov.au/Details/F2020L00552",
        "https://www.york.ac.uk/teaching/cws/wws/webpage1.html",
    ],
)
@pytest.mark.parametrize(
    "parser", [ReadabilityParser(), NewsPleaseParser(), CombinedParser()]
)
@pytest.mark.filterwarnings("ignore::urllib3.exceptions.InsecureRequestWarning")
def test_parse(url: str, parser: HTMLParser) -> None:
    """
    Test that the parser can parse the URL without raising an exception.

    :param url: URL of web page
    :param parser: HTML parser
    """

    input = ParserInput.parse_obj(
        {
            "document_id": "test_id",
            "document_metadata": {
                "publication_ts": "2013-01-01T00:00:00",
                "name": "Dummy Name",
                "description": "description",
                "source_url": "http://existing.com",
                "download_url": None,
                "url": None,
                "md5_sum": None,
                "type": "EU Decision",
                "source": "CCLW",
                "import_id": "TESTCCLW.executive.4.4",
                "family_import_id": "TESTCCLW.family.4.0",
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
            },
            "document_source_url": url,
            "document_cdn_object": None,
            "document_md5_sum": None,
            "document_name": "test_html",
            "document_description": "test_html_description",
            "document_content_type": "text/html",
            "document_slug": "YYY",
        }
    )

    parser_result = parser.parse(input)

    assert isinstance(parser_result, ParserOutput)
    assert parser_result != parser._get_empty_response(input)
