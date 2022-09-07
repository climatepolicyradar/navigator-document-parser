import logging

import requests
from playwright.sync_api import sync_playwright
from playwright.sync_api._generated import Playwright

from src.newsplease import NewsPleaseParser
from src.readability import ReadabilityParser
from src.base import HTMLParser, ParsedHTML
from src.config import MIN_NO_LINES_FOR_VALID_TEXT

logger = logging.getLogger(__name__)


class CombinedParser(HTMLParser):
    """
    Runs the NewsPlease parser on the given URL. If any paragraph is longer than `max_paragraph_words` or the NewsPlease output is empty, it falls back to the Readability parser.

    This has been created as generally NewsPlease is the best parser, but sometimes it pulls paragraphs together, resulting in very long blocks which will be harder to do things with downstream.
    Readability is better at paragraph splitting in these cases, so when NewsPlease creates a long paragraph, we fall back to Readability.

    """

    def __init__(self, max_paragraph_words: int = 500) -> None:
        """Initialise combined parser

        Keyword Arguments:
            max_paragraph_words -- if the longest paragraph has more than this number of words, the parser falls back to the Readability parser (default: {500})
        """
        super().__init__()
        self._max_paragraph_words = max_paragraph_words

    @property
    def name(self) -> str:
        return "combined"

    def parse_html(self, html: str, url: str) -> ParsedHTML:
        """Parse HTML using the better option between NewsPlease and Readability.
        NewsPlease is used unless it returns an empty response or combines paragraphs into paragraphs that we consider too long,
        based on the number of words in them.

        Arguments:
            html -- HTML to parse
            url -- url of web page

        Returns:
            Parsed HTML
        """
        newsplease_result = NewsPleaseParser().parse_html(html, url)

        if len(newsplease_result.text_by_line) == 0:
            return ReadabilityParser().parse_html(html, url)

        if (
            max(
                len(paragraph.split(" "))
                for paragraph in newsplease_result.text_by_line
            )
            > self._max_paragraph_words
        ):
            return ReadabilityParser().parse_html(html, url)

        return newsplease_result

    def parse(self, url: str) -> ParsedHTML:
        """Parse web page using the better option between NewsPlease and Readability. If requests fails to capture HTML that looks like a full web page,
        it falls back to using a headless browser with JS enabled.

        Arguments:
            url -- URL of web page

        Returns:
            Parsed HTML
        """

        # TODO: set timeout and headers in config
        requests_response = requests.get(url, verify=False, allow_redirects=True)

        parsed_html = self.parse_html(requests_response.text, url)

        # If there isn't enough text and there's a `<noscript>` tag in the HTML, try again with JS enabled
        if (len(parsed_html.text_by_line) < MIN_NO_LINES_FOR_VALID_TEXT) and (
            "<noscript>" in requests_response.text
        ):
            with sync_playwright() as playwright:
                html_playwright = self._get_html_with_js_enabled(playwright, url)
                parsed_html_playwright = self.parse_html(html_playwright, url)

            return parsed_html_playwright

        return parsed_html

    def _get_html_with_js_enabled(self, playwright: Playwright, url: str) -> str:
        """Get HTML of a web page using a headless browser with JS enabled

        Arguments:
            playwright -- playwright context manager
            url -- URL of web page

        Returns:
            HTML string
        """

        browser = playwright.chromium.launch()
        context = browser.new_context()
        page = context.new_page()
        page.goto(url)
        html = page.content()
        browser.close()

        return html
