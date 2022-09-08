"""Parser using python-readability library: https://github.com/buriy/python-readability"""

import logging
from typing import List
import re

import requests
from readability import Document
import bleach

from src.config import MIN_NO_LINES_FOR_VALID_TEXT, HTTP_REQUEST_TIMEOUT
from src.base import HTMLParser, ParsedHTML

logger = logging.getLogger(__name__)


class ReadabilityParser(HTMLParser):
    """HTML parser which uses the python-readability library."""

    def __init__(self) -> None:
        super().__init__()

    @property
    def name(self) -> str:
        """Return parser name"""
        return "readability"

    def parse(self, url: str) -> ParsedHTML:
        """Parse web page using readability.

        Arguments:
            url -- URL of web page

        Returns:
            Parsed HTML
        """

        try:
            response = requests.get(
                url, verify=False, allow_redirects=True, timeout=HTTP_REQUEST_TIMEOUT
            )
        except Exception as e:
            logger.error(f"Could not fetch {url}: {e}")
            return self._get_empty_response(url)

        if response.status_code != 200:
            return self._get_empty_response(url)

        return self.parse_html(html=response.text, url=url)

    def parse_html(self, html: str, url: str) -> ParsedHTML:
        """Parse HTML using readability

        Arguments:
            html -- HTML string to parse
            url -- URL of  web page

        Returns:
            Parsed HTML
        """

        readability_doc = Document(html)
        title = readability_doc.title()
        text = readability_doc.summary()
        text_html_stripped = bleach.clean(text, tags=[], strip=True)
        text_by_line = [
            line.strip() for line in text_html_stripped.split("\n") if line.strip()
        ]
        text_by_line = self._combine_bullet_lines_with_next(text_by_line)
        has_valid_text = len(text_by_line) >= MIN_NO_LINES_FOR_VALID_TEXT

        # Readability doesn't provide a date
        return ParsedHTML(
            title=title,
            url=url,
            text_by_line=text_by_line,
            date=None,
            has_valid_text=has_valid_text,
        )

    @staticmethod
    def _combine_bullet_lines_with_next(lines: List[str]) -> List[str]:
        """Iterate through all lines of text. If a line is a bullet or numbered list heading (e.g. (1), 1., i.), then combine it with the next line."""

        list_header_regex = [
            r"([\divxIVX]+\.)+",  # dotted number or roman numeral
            r"(\([\divxIVX]+\))+",  # parenthesized number or roman numeral
            r"[*•\-\–\+]",  # bullets
            r"([a-zA-Z]+\.)+",  # dotted abc
            r"(\([a-zA-Z]+\))+",  # parenthesized abc
        ]

        idx = 0

        while idx < len(lines) - 1:
            if any(re.match(regex, lines[idx].strip()) for regex in list_header_regex):
                lines[idx] = lines[idx].strip() + " " + lines[idx + 1].strip()
                lines[idx + 1] = ""
                idx += 1

            idx += 1

        # strip empty lines
        return [line for line in lines if line]
