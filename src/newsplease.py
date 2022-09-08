import logging

from newsplease import NewsPlease

from src.base import HTMLParser, ParsedHTML
from src.config import MIN_NO_LINES_FOR_VALID_TEXT

logger = logging.getLogger(__name__)


class NewsPleaseParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()

    @property
    def name(self) -> str:
        return "newsplease"

    def parse_html(self, html: str, url: str) -> ParsedHTML:
        """Parse HTML using newsplease

        Arguments:
            html -- HTML string to parse
            url -- URL of web page

        Returns:
            Parsed HTML
        """

        try:
            article = NewsPlease.from_html(html=html, url=url, fetch_images=False)
        except Exception as e:
            logger.error(f"Failed to parse {url}: {e}")
            return self._get_empty_response(url)

        return self._newsplease_article_to_parsed_html(article, url)

    def parse(self, url: str) -> ParsedHTML:
        """Parse website using newsplease

        Arguments:
            url -- URL of web page

        Returns:
            Parsed HTML
        """

        try:
            article = NewsPlease.from_url(url, timeout=30)
        except Exception as e:
            logger.error(f"Failed to parse {url}: {e}")
            return self._get_empty_response(url)

        return self._newsplease_article_to_parsed_html(article, url)

    def _newsplease_article_to_parsed_html(
        self, newsplease_article, url: str
    ) -> ParsedHTML:
        """Convert a newsplease article to parsed HTML. Returns an empty response if the article contains no text.

        Arguments:
            newsplease_article -- article returned by `NewsPlease.from_url` or `NewsPlease.from_html`
            url -- URL of web page

        Returns:
            Parsed HTML
        """

        text = newsplease_article.maintext

        if not text:
            return self._get_empty_response(url)

        text_by_line = text.split("\n")
        has_valid_text = len(text_by_line) >= MIN_NO_LINES_FOR_VALID_TEXT

        return ParsedHTML(
            title=newsplease_article.title,
            url=newsplease_article.url,
            text_by_line=text_by_line,
            date=newsplease_article.date_publish,  # We also have access to the modified and downloaded dates in the class,
            has_valid_text=has_valid_text,
        )
