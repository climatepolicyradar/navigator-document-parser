import logging

from trafilatura import bare_extraction, fetch_url

from src.config import MIN_NO_LINES_FOR_VALID_TEXT
from src.base import HTMLParser, ParsedHTML

logger = logging.getLogger(__name__)


class TrafilaturaParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()

    @property
    def name(self) -> str:
        return "trafilatura"

    def parse(self, url: str) -> ParsedHTML:
        """Parse a URL using trafilatura

        Arguments:
            url -- URL to parse

        Returns:
            parsed HTML
        """

        # TODO: is it secure to disable SSL certificate checking? It looks like we'll need to in order to get data from certain country websites.
        document_html = fetch_url(url, decode=True, no_ssl=True)

        if not document_html:
            logger.warning(f"Could not fetch {url}")
            return self._get_empty_response(url)

        # Ignoring type below because the result of `fetch_url` is a string when `decode=True`
        return self.parse_html(document_html, url)  # type: ignore

    def parse_html(self, html: str, url: str) -> ParsedHTML:
        """Parse HTML using trafilatura

        Arguments:
            html -- string of HTML to parse
            url -- URL of the web page, stored in the output object

        Returns:
            parsed HTML
        """

        try:
            # TODO: decide whether we want to keep formatting. Otherwise disable `include_formatting` flag.
            extracted_dict = bare_extraction(
                html,
                include_tables=False,
                include_formatting=True,
            )
        except Exception as e:
            logger.error(f"Parsing url {url} failed with exception: {e}")
            return self._get_empty_response(url)

        if not extracted_dict:
            logger.error(f"Empty response extracted from {url}")
            return self._get_empty_response(url)

        text_by_line = extracted_dict["text"].split("\n")
        has_valid_text = len(text_by_line) >= MIN_NO_LINES_FOR_VALID_TEXT

        return ParsedHTML(
            title=extracted_dict["title"],
            url=url,
            description=extracted_dict["description"],
            date=extracted_dict["date"],
            text_by_line=text_by_line,
            has_valid_text=has_valid_text,
        )
