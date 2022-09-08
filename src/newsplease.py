"""Parser using news-please library: https://github.com/fhamborg/news-please"""

import logging

from newsplease import NewsPlease

from src.base import HTMLParser, HTMLParserOutput
from src.config import MIN_NO_LINES_FOR_VALID_TEXT, HTTP_REQUEST_TIMEOUT

logger = logging.getLogger(__name__)


class NewsPleaseParser(HTMLParser):
    """HTML parser which uses the news-please library."""

    def __init__(self) -> None:
        super().__init__()

    @property
    def name(self) -> str:
        """Return parser name"""
        return "newsplease"

    def parse_html(self, html: str, url: str) -> HTMLParserOutput:
        """
        Parse HTML using newsplease.

        :param html: HTML string to parse
        :param url: URL of web page

        :return ParsedHTML: parsed HTML
        """

        try:
            article = NewsPlease.from_html(html=html, url=url, fetch_images=False)
        except Exception as e:
            logger.error(f"Failed to parse {url}: {e}")
            return self._get_empty_response(url)

        return self._newsplease_article_to_parsed_html(article, url)

    def parse(self, url: str) -> HTMLParserOutput:
        """
        Parse website using newsplease

        :param url: URL of web page

        :return ParsedHTML: parsed HTML
        """

        try:
            article = NewsPlease.from_url(url, timeout=HTTP_REQUEST_TIMEOUT)
        except Exception as e:
            logger.error(f"Failed to parse {url}: {e}")
            return self._get_empty_response(url)

        return self._newsplease_article_to_parsed_html(article, url)

    def _newsplease_article_to_parsed_html(
        self, newsplease_article, url: str
    ) -> HTMLParserOutput:
        """
        Convert a newsplease article to parsed HTML. Returns an empty response if the article contains no text.

        :param newsplease_article: article returned by `NewsPlease.from_url` or `NewsPlease.from_html`
        :param url: URL of web page

        :return ParsedHTML: parsed HTML
        """

        text = newsplease_article.maintext

        if not text:
            return self._get_empty_response(url)

        text_by_line = text.split("\n")
        has_valid_text = len(text_by_line) >= MIN_NO_LINES_FOR_VALID_TEXT

        return HTMLParserOutput(
            title=newsplease_article.title,
            url=newsplease_article.url,
            text_by_line=text_by_line,
            date=newsplease_article.date_publish,  # We also have access to the modified and downloaded dates in the class
            has_valid_text=has_valid_text,
        )
