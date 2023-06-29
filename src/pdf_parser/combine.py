from typing import List, Tuple

import logging
from layoutparser import Rectangle

from src.base import PDFData, BlockType, PDFTextBlock, PDFPageMetadata
from src.config import BLOCK_OVERLAP_THRESHOLD
from src.pdf_parser.google_ai import (
    get_google_ai_text_blocks,
    PDFPage,
    GoogleTextBlockContent,
)
from src.pdf_parser.layout import LayoutParserWrapper, get_layout_parser_blocks

_LOGGER = logging.getLogger(__file__)


def rectangle_to_coord(rectangle: Rectangle) -> List[Tuple[float, float]]:
    """Converts a layout parser rectangle to a list of coordinates.

    The coordinates represent the rectangle as (x1, y1), (x2, y2).
    Where x1, y1 is the bottom left corner and x2, y2 is the top right corner.
    """
    return [(rectangle.x_1, rectangle.y_1), (rectangle.x_2, rectangle.y_2)]


def assign_block_type(
    parsed_document_pages: list[PDFPage],
    lp_obj: LayoutParserWrapper,
    document_md5sum: str,
) -> PDFData:
    """The google document ai api has many good features, however it does not support text block type detection.

    For example ‘Table’ or ‘Figure’.
    This is necessary as we are may want to filter these out at a later date etc.

    To solve this problem we want to use layout parser to detect types and boxes in the documents and assign the
    types detected in layout parser to text blocks identified from the google ai api.
    """
    document_text_blocks = []
    document_pages_metadata = []

    for page in parsed_document_pages:
        layout_parser_blocks: list = get_layout_parser_blocks(
            page.extracted_content.pages[0].image.content, lp_obj
        )

        google_ai_layout_coords: list[
            GoogleTextBlockContent
        ] = get_google_ai_text_blocks(
            page=page.extracted_content.pages[0],
            document_text=page.extracted_content.text,
        )

        for layout_block in layout_parser_blocks:
            for google_ai_block in google_ai_layout_coords:
                block_type = BlockType.AMBIGUOUS
                block_confidence = 0.0
                if (
                    layout_block.block.intersect(google_ai_block.coordinates).area
                    / layout_block.block.area
                    > BLOCK_OVERLAP_THRESHOLD
                ):
                    block_type = BlockType(layout_block.type)
                    block_confidence = layout_block.score

                # FIXME The type for languages is a string so will take the first.
                #   [lang.language_code for lang in page.detected_languages]

                document_text_blocks.append(
                    PDFTextBlock(
                        coords=rectangle_to_coord(google_ai_block.coordinates),
                        page_number=page.page_number,
                        text_block_id=int(len(document_text_blocks)) + 1,
                        type=block_type,
                        type_confidence=block_confidence,
                        text=[google_ai_block.text],
                        language=page.extracted_content.pages[0]
                        .detected_languages[0]
                        .language_code,
                    )
                )

        document_pages_metadata.append(
            PDFPageMetadata(
                page_number=page.page_number,
                dimensions=(
                    page.extracted_content.pages[0].layout.bounding_poly.vertices[2].x,
                    page.extracted_content.pages[0].layout.bounding_poly.vertices[2].y,
                ),
            )
        )

    return PDFData(
        page_metadata=document_pages_metadata,
        text_blocks=document_text_blocks,
        md5sum=document_md5sum,
    )
