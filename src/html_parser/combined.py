"""A combined parser which uses both readability and newsplease to parse HTML."""

import logging

import requests
from playwright.sync_api import sync_playwright
from playwright.sync_api._generated import Playwright

from src.html_parser.newsplease import NewsPleaseParser
from src.html_parser.readability import ReadabilityParser
from src.base import HTMLParser, ParserInput, ParserOutput
from src.config import (
    HTML_MIN_NO_LINES_FOR_VALID_TEXT,
    HTML_HTTP_REQUEST_TIMEOUT,
    HTML_MAX_PARAGRAPH_LENGTH_WORDS,
)

logger = logging.getLogger(__name__)


class CombinedParser(HTMLParser):
    """
    Runs the NewsPlease parser on the given URL. If any paragraph is longer than `max_paragraph_words` or the NewsPlease output is empty, it falls back to the Readability parser.

    This has been created as generally NewsPlease is the best parser, but sometimes it pulls paragraphs together, resulting in very long blocks which will be harder to do things with downstream.
    Readability is better at paragraph splitting in these cases, so when NewsPlease creates a long paragraph, we fall back to Readability.

    """

    def __init__(
        self, max_paragraph_words: int = HTML_MAX_PARAGRAPH_LENGTH_WORDS
    ) -> None:
        """
        Initialise combined parser

        Keyword Arguments:
            max_paragraph_words -- if the longest paragraph has more than this number of words, the parser falls back to the Readability parser (default: {500})
        """
        super().__init__()
        self._max_paragraph_words = max_paragraph_words

    @property
    def name(self) -> str:
        """Return parser name"""
        return "combined"

    def parse_html(self, html: str, input: ParserInput) -> ParserOutput:
        """
        Parse HTML using the better option between NewsPlease and Readability.

        NewsPlease is used unless it returns an empty response or combines paragraphs into paragraphs that we consider too long,
        based on the number of words in them.

        :param html: HTML to parse
        :param url: url of web page

        :return ParsedHTML: Parsed HTML
        """
        newsplease_result = NewsPleaseParser().parse_html(html, input)

        if len(newsplease_result.text_blocks) == 0:
            return ReadabilityParser().parse_html(html, input)

        if (
            max(
                len(paragraph.to_string().split(" "))
                for paragraph in newsplease_result.text_blocks
            )
            > self._max_paragraph_words
        ):
            return ReadabilityParser().parse_html(html, input)

        return newsplease_result

    def parse(self, input: ParserInput) -> ParserOutput:
        """
        Parse web page using the better option between NewsPlease and Readability. If requests fails to capture HTML that looks like a full web page, it falls back to using a headless browser with JS enabled.

        :param url: URL of web page

        :return ParsedHTML: parsed HTML
        """

        try:
            requests_response = requests.get(
                input.document_url,
                verify=False,
                allow_redirects=True,
                timeout=HTML_HTTP_REQUEST_TIMEOUT,
            )
        except Exception as e:
            logger.error(
                f"Could not fetch {input.document_url} for {input.document_id}: {e}"
            )
            return self._get_empty_response(input)

        try:
            parsed_html = self.parse_html(requests_response.text, input)
        except Exception as e:
            logger.error(
                f"Could not parse {input.document_url} for {input.document_id}: {e}"
            )
            return self._get_empty_response(input)

        # If there isn't enough text and there's a `<noscript>` tag in the HTML, try again with JS enabled
        if (len(parsed_html.text_blocks) < HTML_MIN_NO_LINES_FOR_VALID_TEXT) and (
            "<noscript>" in requests_response.text
        ):
            logger.info(
                f"Falling back to JS-enabled browser for {input.document_id} ({input.document_url})"
            )
            try:
                with sync_playwright() as playwright:
                    html_playwright = self._get_html_with_js_enabled(
                        playwright, input.document_url
                    )
                    parsed_html_playwright = self.parse_html(html_playwright, input)

                return parsed_html_playwright
            except Exception as e:
                logger.error(
                    f"Could not fetch {input.document_url} for {input.document_id}: {e}"
                )
                return self._get_empty_response(input)

        return parsed_html

    def _get_html_with_js_enabled(self, playwright: Playwright, url: str) -> str:
        """
        Get HTML of a web page using a headless browser with JS enabled

        :param playwright: playwright context manager
        :param url: URL of web page

        :return str: HTML string
        """

        browser = playwright.chromium.launch()
        context = browser.new_context()
        page = context.new_page()
        page.goto(url)
        html = page.content()
        browser.close()

        return html
