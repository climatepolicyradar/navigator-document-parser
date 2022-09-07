from typing import Optional, List
from abc import ABC
from datetime import date

from pydantic import BaseModel
from langdetect import detect
from langdetect import DetectorFactory


class ParsedHTML(BaseModel):
    title: Optional[str]
    url: str
    text_by_line: List[str]
    date: Optional[date]
    has_valid_text: bool
    language: Optional[str] = None
    translated: bool = False

    def detect_language(self) -> "ParsedHTML":
        """Detect language of the text and set the language attribute. Return an instance of ParsedHTML with the language attribute set.
        TODO: we could detect a language per element instead. Are we safe to assume that a website is written in only one language?
        """

        # language detection is not deterministic, so we need to set a seed
        DetectorFactory.seed = 0

        if self.text_by_line:
            self.language = detect(" ".join(self.text_by_line))

        return self


class HTMLParser(ABC):
    """Base class for an HTML parser."""

    @property
    def name(self) -> str:
        raise NotImplementedError()

    def parse_html(self, html: str, url: str) -> ParsedHTML:
        raise NotImplementedError()

    def parse(self, url: str) -> ParsedHTML:
        raise NotImplementedError()

    def _get_empty_response(self, url) -> ParsedHTML:
        """Return ParsedHTML object with empty fields."""
        return ParsedHTML(
            title="",
            url=url,
            description=None,
            date=None,
            text_by_line=[],
            has_valid_text=False,
        )
