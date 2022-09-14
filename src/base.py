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


class ParserInput(BaseModel):
    """Base class for input to a parser."""

    id: str
    url: AnyHttpUrl
    content_type: str
    document_slug: str


class ParserOutput(BaseModel):
    """Base class for an output to a parser."""

    id: str
    url: AnyHttpUrl
    languages: Optional[List[str]] = None
    translated: bool = False


class HTMLParserOutput(ParserOutput):
    """Base class for the output of an HTML parser."""

    title: Optional[str]
    text_by_line: List[str]
    date: Optional[date]
    has_valid_text: bool

    def set_languages(self) -> "HTMLParserOutput":
        """
        Detect language of the text and set the language attribute. Return an instance of ParsedHTML with the language attribute set.

        TODO: we assume an HTML page contains only one language here. Do we want to try to detect chunks of text with potentially differing languages?
        """

        # language detection is not deterministic, so we need to set a seed
        DetectorFactory.seed = 0

        if self.text_by_line:
            self.languages = [detect(" ".join(self.text_by_line))]

        return self


class PDFBlockType(str, Enum):
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


class PDFTextBlock(BaseModel):
    """
    Represents an individual text block on a page.

    Stores the text and positional information for a single
    text block extracted from a document.

    Attributes:
        text: list of text lines contained in the text block
        text_block_id: unique identifier for the text block
        coords: list of coordinates of the vertices defining the boundary of the text block.
           Each coordinate is a tuple in the format (x, y). (0, 0) is at the top left corner of
           the page, and the positive x- and y- directions are right and down.
        type: predicted type of the text block
        type_confidence: confidence score of the text block being of the predicted type
        language: language of the text block, 2-letter ISO code, optional.
    """

    text: List[str]
    text_block_id: str
    coords: List[Tuple[float, float]]
    type: PDFBlockType
    type_confidence: float = Field(ge=0, le=1)
    language: Optional[
        str
    ] = None  # TODO: validate this against a list of language ISO codes

    def to_string(self) -> str:
        """Returns the lines in a text block as a string with the lines separated by spaces."""

        return " ".join([line.strip() for line in self.text])

    @classmethod
    def from_layoutparser(
        cls, text_block: lp_elements.TextBlock, text_block_id: str
    ) -> "PDFTextBlock":
        """
        Create a TextBlock from a LayoutParser TextBlock.

        :param text_block: LayoutParser TextBlock
        :param text_block_id: ID to use for the resulting TextBlock.
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
        )


class PDFPage(BaseModel):
    """
    Represents a page in a document.

    All text blocks on a page are contained within a Page object. Also, the dimensions of the page can
    be specified.

    :attribute text_blocks: List of text blocks contained in the document
    :attribute dimensions: The dimensions of the page as a tuple in the format (x, y), where x is horizontal and y is vertical dimension.
    :attribute page_number: Unique id of the page, e.g. page number starting at 0.
    """

    text_blocks: List[PDFTextBlock]
    dimensions: Tuple[float, float]
    page_number: int = Field(ge=0)

    def to_string(self) -> str:
        """Return the text blocks in the page as a string"""

        page_text = [text_block.to_string().strip() for text_block in self.text_blocks]

        return "\n".join(page_text)

    @classmethod
    def from_layoutparser(
        cls, layout: lp_elements.Layout, page_num: int, dimensions: Tuple[float, float]
    ) -> "PDFPage":
        """
        Create a Page from a LayoutParser Layout.

        :param layout: LayoutParser Layout
        :param page_num: page number, 0-indexed
        :param dimensions: (width, height) in pixels
        :return Page:
        """
        text_blocks = []

        for block_idx, lp_text_block in enumerate(layout._blocks):
            block_id = f"p{page_num}_b{block_idx}"
            text_blocks.append(PDFTextBlock.from_layoutparser(lp_text_block, block_id))

        return PDFPage(
            text_blocks=text_blocks,
            dimensions=dimensions,
            page_number=page_num,
        )


class PDFParserOutput(BaseModel):
    """
    Represents a document and associated pages and text blocks.

    Stores all of the pages that are contained in a document.

    :attribute pages: List of pages contained in the document
    :attribute filename: Name of the PDF file, without extension
    :attribute md5hash: md5sum of PDF content
    :attribute language: list of 2-letter ISO language codes, optional. If null, the OCR processor didn't support language detection
    """

    id: str
    pages: List[PDFPage]  # List of textblocks in the document
    document_slug: str  # for better links to the frontend hopefully soon
    md5hash: str  # MD5 hash of the pdf file
    languages: Optional[
        List[str]
    ] = None  # TODO: validate this against a list of language ISO codes

    def set_languages(self, min_language_proportion: float = 0.4):
        """
        Store the document languages attribute as part of the object by getting all languages with proportion above `min_language_proportion`.

        :attribute min_language_proportion: Minimum proportion of text blocks in a language for it to be considered a language of the document.
        """

        all_text_block_languages = [
            text_block.language
            for page in self.pages
            for text_block in page.text_blocks
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
            title="",
            url=input.url,
            date=None,
            text_by_line=[],
            has_valid_text=False,
        )
