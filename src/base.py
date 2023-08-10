"""Base classes for parsing."""
import logging.config
from abc import ABC, abstractmethod
from collections import Counter
from datetime import date
from enum import Enum
from typing import Optional, Sequence, Tuple, List, Any

from azure.ai.formrecognizer import Point
from langdetect import DetectorFactory, detect
from pydantic import BaseModel, Field, root_validator

logger = logging.getLogger(__name__)

CONTENT_TYPE_HTML = "text/html"
CONTENT_TYPE_PDF = "application/pdf"


class PDFPage(BaseModel):
    """Represents a batch of pages from a PDF document."""

    page_number: int
    extracted_content: Any


class BlockType(str, Enum):
    """
    List of possible block types.
    """

    TEXT = "Text"
    TITLE = "Title"
    LIST = "List"
    TABLE = "Table"
    FIGURE = "Figure"
    INFERRED = "Inferred from gaps"
    AMBIGUOUS = "Ambiguous"
    GOOGLE_BLOCK = "Google Text Block"


class TextBlock(BaseModel):
    """
    Base class for a text block.
    :attribute text: list of text lines contained in the text block
    :attribute text_block_id: unique identifier for the text block
    :attribute language: language of the text block. 2-letter ISO code, optional.
    :attribute type: predicted type of the text block
    :attribute type_confidence: confidence score of the text block being of the predicted type
    """

    text: List[str]
    text_block_id: str
    language: Optional[
        str
    ] = None
    # FIXME: Setting as string for now as we don't have a types for all the new options
    type: str  # BlockType
    type_confidence: float = Field(ge=0, le=1)

    def to_string(self) -> str:
        """Returns the lines in a text block as a string with the lines separated by spaces."""

        return " ".join([line.strip() for line in self.text])


class HTMLTextBlock(TextBlock):
    """
    Text block parsed from an HTML document. Type is set to "Text" with a confidence of 1.0 by default,
    as we do not predict types for text blocks parsed from HTML.
    """

    type: BlockType = BlockType.TEXT
    type_confidence: float = 1.0


class PDFTextBlock(TextBlock):
    """
    Text block parsed from a PDF document.
    Stores the text and positional information for a single text block extracted from a document.
    :attribute coords: list of coordinates of the vertices defining the boundary of the text block.
        Each coordinate is a tuple in the format (x, y). (0, 0) is at the top left corner of
        the page, and the positive x- and y- directions are right and down.
    :attribute page_number: page number of the page containing the text block.
    """

    coords: List[Tuple[float, float]]
    page_number: int = Field(ge=0)

    def to_string(self) -> str:
        """Returns the lines in a text block as a string with the lines separated by spaces."""

        return " ".join([line.strip() for line in self.text])


class ParserInput(BaseModel):
    """Base class for input to a parser."""

    document_id: str
    document_metadata: dict
    document_name: str
    document_description: str
    document_source_url: Optional[str]
    document_cdn_object: Optional[str]
    document_content_type: Optional[str]
    document_md5_sum: Optional[str]
    document_slug: str


class HTMLData(BaseModel):
    """Set of metadata specific to HTML documents."""

    detected_title: Optional[str]
    detected_date: Optional[date]
    has_valid_text: bool
    text_blocks: Sequence[HTMLTextBlock]


class PDFPageMetadata(BaseModel):
    """Set of metadata for a single page of a PDF document."""

    page_number: int = Field(ge=0)
    dimensions: Tuple[float, float]


class BoundingRegion(BaseModel):
    """Region of a cell in a table."""

    page_number: int
    polygon: List[Point]


class TableCell(BaseModel):
    """Cell of a table."""
    # TODO create cell type enum
    # TODO potentially enforce bounding regions to be one region per page
    cell_type: str
    row_index: int
    column_index: int
    row_span: int
    column_span: int
    content: str
    bounding_regions: List[BoundingRegion]


class PDFTableBlock(BaseModel):
    """Table block parsed form a PDF document.
    Stores the text and positional information for a single table block extracted from a document.
    """

    table_id: str
    row_count: int
    column_count: int
    cells: List[TableCell]


class PDFData(BaseModel):
    """Set of metadata unique to PDF documents."""

    page_metadata: Sequence[PDFPageMetadata]
    md5sum: str
    text_blocks: Optional[Sequence[PDFTextBlock]] = None
    table_blocks: Optional[Sequence[PDFTableBlock]] = None


class ParserOutput(BaseModel):
    """Base class for an output to a parser."""

    document_id: str
    document_metadata: dict
    document_name: str
    document_description: str
    document_source_url: Optional[str]
    document_cdn_object: Optional[str]
    document_content_type: Optional[str]
    document_md5_sum: Optional[str]
    document_slug: str

    languages: Optional[Sequence[str]] = None
    translated: bool = False
    html_data: Optional[HTMLData] = None
    pdf_data: Optional[PDFData] = None

    @root_validator
    def check_html_pdf_metadata(cls, values):
        """Check that html_data is set if content_type is HTML, or pdf_data is set if content_type is PDF."""
        if (
            values["document_content_type"] == CONTENT_TYPE_HTML
            and values["html_data"] is None
        ):
            raise ValueError("html_metadata must be set for HTML documents")

        if (
            values["document_content_type"] == CONTENT_TYPE_PDF
            and values["pdf_data"] is None
        ):
            raise ValueError("pdf_data must be set for PDF documents")

        if values["document_content_type"] is None and (
            values["html_data"] is not None or values["pdf_data"] is not None
        ):
            raise ValueError(
                "html_metadata and pdf_metadata must be null for documents with no content type."
            )

        return values

    @property
    def text_blocks(self) -> Sequence[TextBlock]:
        """
        Return the text blocks in the document. These could differ in format depending on the content type.
        :return: Sequence[TextBlock]
        """

        if self.document_content_type == CONTENT_TYPE_HTML:
            return self.html_data.text_blocks
        elif self.document_content_type == CONTENT_TYPE_PDF:
            return self.pdf_data.text_blocks

    def to_string(self) -> str:
        """Return the text blocks in the parser output as a string"""

        return " ".join(
            [text_block.to_string().strip() for text_block in self.text_blocks]
        )

    def detect_and_set_languages(self) -> "ParserOutput":
        """
        Detect language of the text and set the language attribute. Return an instance of ParserOutput with the language attribute set.

        Assumes that a document only has one language.
        """

        # FIXME: We can remove this now as this api doesn't support language detection
        if self.document_content_type != CONTENT_TYPE_HTML:
            logger.warning(
                "Language detection should not be required for non-HTML documents, but it has been run on one. "
                "This will overwrite any document languages detected via other means, e.g. OCR. "
            )

        # language detection is not deterministic, so we need to set a seed
        DetectorFactory.seed = 0

        if len(self.text_blocks) > 0:
            detected_language = detect(self.to_string())
            self.languages = [detected_language]
            for text_block in self.text_blocks:
                text_block.language = detected_language

        return self

    def set_document_languages_from_text_blocks(
        self, min_language_proportion: float = 0.4
    ):
        """
        Store the document languages attribute as part of the object by getting all languages with proportion
        above `min_language_proportion`. :attribute min_language_proportion: Minimum proportion of text blocks
        in a language for it to be considered a language of the document.
        """

        all_text_block_languages = [
            text_block.language for text_block in self.text_blocks
        ]

        if all([lang is None for lang in all_text_block_languages]):
            self.languages = None

        else:
            lang_counter = Counter(
                [lang for lang in all_text_block_languages if lang is not None]
            )
            self.languages = [
                lang
                for lang, count in lang_counter.items()
                if count / len(all_text_block_languages) > min_language_proportion
            ]

        return self


class HTMLParser(ABC):
    """Base class for an HTML parser."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier for the parser. Can be used if we want to identify the parser that parsed a web page."""
        raise NotImplementedError()

    @abstractmethod
    def parse_html(self, html: str, url: str) -> ParserOutput:
        """Parse an HTML string directly."""
        raise NotImplementedError()

    @abstractmethod
    def parse(self, input: ParserInput) -> ParserOutput:
        """Parse a web page, by fetching the HTML and then parsing it. Implementations will often call `parse_html`."""
        raise NotImplementedError()

    def _get_empty_response(self, input: ParserInput) -> ParserOutput:
        """Return ParsedHTML object with empty fields."""
        return ParserOutput(
            document_id=input.document_id,
            document_metadata=input.document_metadata,
            document_content_type=input.document_content_type,
            document_name=input.document_name,
            document_description=input.document_description,
            document_source_url=input.document_source_url,
            document_cdn_object=input.document_cdn_object,
            document_md5_sum=input.document_md5_sum,
            document_slug=input.document_slug,
            html_data=HTMLData(
                text_blocks=[],
                detected_date=None,
                detected_title="",
                has_valid_text=False,
            ),
        )
