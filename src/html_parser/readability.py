"""Parser using python-readability library: https://github.com/buriy/python-readability"""

import logging
from typing import List
import re

import requests
from readability import Document
import bleach

from src.config import MIN_NO_LINES_FOR_VALID_TEXT, HTTP_REQUEST_TIMEOUT
from src.base import HTMLParser, ParserInput, ParserOutput, HTMLData, HTMLTextBlock

logger = logging.getLogger(__name__)


class ReadabilityParser(HTMLParser):
    """HTML parser which uses the python-readability library."""

    def __init__(self) -> None:
        super().__init__()

    @property
    def name(self) -> str:
        """Return parser name"""
        return "readability"

    def parse(self, input: ParserInput) -> ParserOutput:
        """
        Parse web page using readability.

        :param url: URL of web page

        :return ParsedHTML: parsed HTML
        """

        try:
            response = requests.get(
                input.url,
                verify=False,
                allow_redirects=True,
                timeout=HTTP_REQUEST_TIMEOUT,
            )
        except Exception as e:
            logger.error(f"Could not fetch {input.url} for {input.id}: {e}")
            return self._get_empty_response(input)

        if response.status_code != 200:
            return self._get_empty_response(input)

        return self.parse_html(response.text, input)

    def parse_html(self, html: str, input: ParserInput) -> ParserOutput:
        """Parse HTML using readability

        :param html: HTML string to parse
        :param url: URL of  web page

        :return ParsedHTML: parsed HTML
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

        text_blocks = [
            HTMLTextBlock.parse_obj(
                {
                    "text_block_id": f"b{idx}",
                    "text": [text],
                }
            )
            for idx, text in enumerate(text_by_line)
        ]

        # Readability doesn't provide a date
        return ParserOutput(
            id=input.id,
            content_type=input.content_type,
            document_name=input.document_name,
            document_description=input.document_description,
            url=input.url,
            document_slug=input.document_slug,
            html_data=HTMLData(
                detected_title=title,
                detected_date=None,
                has_valid_text=has_valid_text,
                text_blocks=text_blocks,
            ),
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
