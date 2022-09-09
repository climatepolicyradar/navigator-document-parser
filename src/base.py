"""Base classes for parsing."""

from typing import Optional, List
from abc import ABC
from datetime import date

from pydantic import BaseModel, AnyHttpUrl
from langdetect import detect
from langdetect import DetectorFactory


class HTMLParserInput(BaseModel):
    """Input for the parser."""

    id: str
    url: AnyHttpUrl


class HTMLParserOutput(BaseModel):
    """Base class for the output of an HTML parser."""

    id: str
    url: AnyHttpUrl
    title: Optional[str]
    text_by_line: List[str]
    date: Optional[date]
    has_valid_text: bool
    language: Optional[str] = None
    translated: bool = False

    def detect_language(self) -> "HTMLParserOutput":
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
        """Identifier for the parser. Can be used if we want to identify the parser that parsed a web page."""
        raise NotImplementedError()

    def parse_html(self, html: str, url: str) -> HTMLParserOutput:
        """Parse an HTML string directly."""
        raise NotImplementedError()

    def parse(self, input: HTMLParserInput) -> HTMLParserOutput:
        """Parse a web page, by fetching the HTML and then parsing it. Implementations will often call `parse_html`."""
        raise NotImplementedError()

    def _get_empty_response(self, input: HTMLParserInput) -> HTMLParserOutput:
        """Return ParsedHTML object with empty fields."""
        return HTMLParserOutput(
            id=input.id,
            title="",
            url=input.url,
            date=None,
            text_by_line=[],
            has_valid_text=False,
        )
