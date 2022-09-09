import pytest

from src.base import HTMLParser, HTMLParserInput, HTMLParserOutput
from src.newsplease import NewsPleaseParser
from src.readability import ReadabilityParser
from src.combined import CombinedParser


@pytest.mark.parametrize(
    "url",
    [
        "https://www.legislation.gov.au/Details/F2020L00552",
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

    input = HTMLParserInput.parse_obj({"id": "test_id", "url": url})

    parser_result = parser.parse(input)

    assert isinstance(parser_result, HTMLParserOutput)
    assert parser_result != parser._get_empty_response(input)
