import pytest

from src.base import HTMLParser, ParserInput, ParserOutput
from src.html_parser.newsplease import NewsPleaseParser
from src.html_parser.readability import ReadabilityParser
from src.html_parser.combined import CombinedParser


@pytest.mark.parametrize(
    "url",
    [
        # "https://www.legislation.gov.au/Details/F2020L00552",
        "https://www.bopa.ad/bopa/030040/Pagines/GD20180622_09_07_48.aspx",
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
            "document_metadata": {},
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
