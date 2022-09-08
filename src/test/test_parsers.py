import pytest

from src.base import HTMLParser, ParsedHTML
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
    """Test that the parser can parse the URL without raising an exception.

    Arguments:
        url -- URL of web page
        parser -- parser to test

    Returns:
        None
    """

    parser_result = parser.parse(url)

    assert isinstance(parser_result, ParsedHTML)
