"""Base classes for parsing."""

from enum import Enum
from typing import Optional, List, Tuple
from abc import ABC
from datetime import date

from collections import Counter
from pydantic import BaseModel, AnyHttpUrl, Field
from langdetect import detect
from langdetect import DetectorFactory
import layoutparser.elements as lp_elements


class BlockType(str, Enum):
    """
    List of possible block types from the PubLayNet model.

    https://layout-parser.readthedocs.io/en/latest/notes/modelzoo.html#model-label-map
    """

    TEXT = "Text"
    TITLE = "Title"
    LIST = "List"
    TABLE = "Table"
    FIGURE = "Figure"
    AMBIGUOUS = "Ambiguous"  # TODO: remove this when OCRProcessor._infer_block_type is implemented


class ContentType(str, Enum):
    """List of document content types that can be handled by the parser."""

    HTML = "text/html"
    PDF = "application/pdf"


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
    ] = None  # TODO: validate this against a list of language ISO codes
    type: BlockType
    type_confidence: float = Field(ge=0, le=1)

    def to_string(self) -> str:
        """Returns the lines in a text block as a string with the lines separated by spaces."""

        return " ".join([line.strip() for line in self.text])


class HTMLTextBlock(TextBlock):
    """
    Text block parsed from an HTML document.

    Type is set to "Text" with a confidence of 1.0 by default, as we do not predict types for text blocks parsed from HTML.
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

    @classmethod
    def from_layoutparser(
        cls, text_block: lp_elements.TextBlock, text_block_id: str, page_number: int
    ) -> "PDFTextBlock":
        """
        Create a TextBlock from a LayoutParser TextBlock.

        :param text_block: LayoutParser TextBlock
        :param text_block_id: ID to use for the resulting TextBlock.
        :param page_number: Page number of the text block.
        :raises ValueError: if the LayoutParser TextBlock does not have all of the properties `text`, `coordinates`, `score` and `type`.
        :return TextBlock:
        """

        null_values_of_lp_block = [
            k
            for k, v in {
                "text": text_block.text,
                "coordinates": text_block.coordinates,
                "score": text_block.score,
                "type": text_block.type,
            }.items()
            if v is None
        ]

        if len(null_values_of_lp_block) > 0:
            raise ValueError(
                f"LayoutParser TextBlock has null values: {null_values_of_lp_block}"
            )

        # Convert from a potentially not rectangular quadrilateral to a rectangle.
        # This method does nothing if the text block is already a rectangle.
        text_block = text_block.to_rectangle()

        new_format_coordinates = [
            (text_block.block.x_1, text_block.block.y_1),
            (text_block.block.x_2, text_block.block.y_1),
            (text_block.block.x_2, text_block.block.y_2),
            (text_block.block.x_1, text_block.block.y_2),
        ]

        # Ignoring types below as this method will raise an error if any of these values are None above.
        return PDFTextBlock(
            text=[text_block.text],  # type: ignore
            text_block_id=text_block_id,  # e.g. p0_b3
            coords=new_format_coordinates,  # type: ignore
            type_confidence=text_block.score,  # type: ignore
            type=text_block.type,  # type: ignore
            page_number=page_number,
        )


class ParserInput(BaseModel):
    """Base class for input to a parser."""

    id: str
    url: AnyHttpUrl
    content_type: ContentType
    document_slug: str


class ParserOutput(BaseModel):
    """Base class for an output to a parser."""

    id: str
    url: AnyHttpUrl
    languages: Optional[List[str]] = None
    text_blocks: List[TextBlock]
    translated: bool = False
    document_slug: str  # for better links to the frontend hopefully soon
    content_type: ContentType

    def to_string(self) -> str:
        """Return the text blocks in the parser output as a string"""

        return " ".join(
            [text_block.to_string().strip() for text_block in self.text_blocks]
        )


class PDFPageMetadata(BaseModel):
    """
    Set of metadata for a single page of a PDF document.

    :attribute dimensions: (width, height) of the page in pixels
    """

    page_number: int = Field(ge=0)
    dimensions: Tuple[float, float]


class HTMLParserOutput(ParserOutput):
    """Base class for the output of an HTML parser."""

    title: Optional[str]
    text_blocks: List[HTMLTextBlock]
    date: Optional[date]
    has_valid_text: bool

    def set_languages(self) -> "HTMLParserOutput":
        """
        Detect language of the text and set the language attribute. Return an instance of ParsedHTML with the language attribute set.

        TODO: we assume an HTML page contains only one language here. Do we want to try to detect chunks of text with potentially differing languages?
        """

        # language detection is not deterministic, so we need to set a seed
        DetectorFactory.seed = 0

        if len(self.text_blocks) > 0:
            self.languages = [detect(self.to_string())]

        return self


class PDFParserOutput(ParserOutput):
    """
    Represents a document and associated pages and text blocks.

    Stores all of the pages that are contained in a document.

    :attribute pages: List of pages contained in the document
    :attribute filename: Name of the PDF file, without extension
    :attribute md5sum: md5sum of PDF content
    :attribute language: list of 2-letter ISO language codes, optional. If null, the OCR processor didn't support language detection
    """

    page_metadata: List[PDFPageMetadata]
    text_blocks: List[PDFTextBlock]
    md5sum: str

    def set_languages(self, min_language_proportion: float = 0.4):
        """
        Store the document languages attribute as part of the object by getting all languages with proportion above `min_language_proportion`.

        :attribute min_language_proportion: Minimum proportion of text blocks in a language for it to be considered a language of the document.
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
    def name(self) -> str:
        """Identifier for the parser. Can be used if we want to identify the parser that parsed a web page."""
        raise NotImplementedError()

    def parse_html(self, html: str, url: str) -> HTMLParserOutput:
        """Parse an HTML string directly."""
        raise NotImplementedError()

    def parse(self, input: ParserInput) -> HTMLParserOutput:
        """Parse a web page, by fetching the HTML and then parsing it. Implementations will often call `parse_html`."""
        raise NotImplementedError()

    def _get_empty_response(self, input: ParserInput) -> HTMLParserOutput:
        """Return ParsedHTML object with empty fields."""
        return HTMLParserOutput(
            id=input.id,
            content_type=input.content_type,
            title="",
            url=input.url,
            date=None,
            text_blocks=[],
            document_slug=input.document_slug,
            has_valid_text=False,
        )
