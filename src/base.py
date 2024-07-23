"""Base classes for parsing."""

import logging
import logging.config
from abc import ABC, abstractmethod

from cpr_sdk.parser_models import HTMLData, ParserOutput, ParserInput

logger = logging.getLogger(__name__)


PARSER_METADATA_KEY = "parser_metadata"


class HTMLParser(ABC):
    """Base class for an HTML parser."""

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Identifier for the parser.

        Can be used if we want to identify the parser that parsed a web page.
        """
        raise NotImplementedError()

    @abstractmethod
    def parse_html(self, html: str, url: str) -> ParserOutput:
        """Parse an HTML string directly."""
        raise NotImplementedError()

    @abstractmethod
    def parse(self, input_: ParserInput) -> ParserOutput:
        """
        Parse a web page, by fetching the HTML and then parsing it.

        Implementation will often call `parse_html`.
        """
        raise NotImplementedError()

    def _get_empty_response(self, input_: ParserInput) -> ParserOutput:
        """Return ParsedHTML object with empty fields."""
        return ParserOutput(
            document_id=input_.document_id,
            document_metadata=input_.document_metadata,
            document_content_type=input_.document_content_type,
            document_name=input_.document_name,
            document_description=input_.document_description,
            document_source_url=input_.document_source_url,
            document_cdn_object=input_.document_cdn_object,
            document_md5_sum=input_.document_md5_sum,
            document_slug=input_.document_slug,
            html_data=HTMLData(
                text_blocks=[],
                detected_date=None,
                detected_title="",
                has_valid_text=False,
            ),
        )
