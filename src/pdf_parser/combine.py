from typing import List, Tuple

import google
from google.cloud import documentai
from layoutparser import Rectangle

from src.base import PDFData, BlockType, PDFTextBlock, PDFPageMetadata
from src.pdf_parser.layout import LayoutParserWrapper


def layout_to_text(layout: documentai.Document.Page.Layout, text: str) -> str:
    """
    Document AI identifies text in different parts of the document by their offsets in the entirety of the
    document's text. This function converts offsets to a string.

    If a text segment spans several lines, it will be stored in different text segments.
    """
    response = ""
    for segment in layout.text_anchor.text_segments:
        start_index = int(segment.start_index)
        end_index = int(segment.end_index)
        response += text[start_index:end_index]
    return response


def rectangle_to_coord(rectangle: Rectangle) -> List[Tuple[float, float]]:
    """Converts a layout parser rectangle to a list of coordinates.

    The coordinates represent the rectangle as (x1, y1), (x2, y2).
    Where x1, y1 is the bottom left corner and x2, y2 is the top right corner.
    """
    return [
        (rectangle.x_1, rectangle.y_1),
        (rectangle.x_2, rectangle.y_2)
    ]


def assign_block_type(document: google.cloud.documentai_v1.Document, lp_obj: LayoutParserWrapper) -> PDFData:
    """The google document ai api has many good features, however it does not support text block type detection.

    For example ‘Table’ or ‘Figure’.
    This is necessary as we are may want to filter these out at a later date etc.

    To solve this problem we want to use layout parser to detect types and boxes in the documents and assign the
    types detected in layout parser to text blocks identified from the google ai api.
    """
    document_text_blocks = []
    document_pages_metadata = []
    document_md5sum = document.md5_checksum

    for page in document.pages:
        layout = lp_obj.get_layout(page.image.content)
        layout_coords = [block.block for block in layout._blocks]

        page_vertices = page.layout.bounding_poly.vertices[2]

        google_ai_blocks = [
            block.layout.bounding_poly.normalized_vertices
            for block in page.blocks
        ]

        google_ai_coords = [
            Rectangle(x_1=block[0].x, y_1=block[0].y, x_2=block[2].x, y_2=block[2].y)
            for block in google_ai_blocks
        ]

        google_ai_coords_scaled = [
            Rectangle(
                x_1=block.x_1*page_vertices.x,
                y_1=block.y_1*page_vertices.y,
                x_2=block.x_2*page_vertices.x,
                y_2=block.y_2*page_vertices.y
            )
            for block in google_ai_coords
        ]

        for layout_block in layout_coords:
            for google_ai_block in google_ai_coords_scaled:

                block_type = BlockType.AMBIGUOUS
                block_confidence = 0.0
                if layout_block.intersect(google_ai_block).area / layout_block.area > 0.7:
                    block_type = BlockType(layout_block.type)
                    block_confidence = layout_block.score

                # FIXME The type for languages is a string so will take the first.
                #   [lang.language_code for lang in page.detected_languages]
                # FIXME: Set threshold from env vars or config

                document_text_blocks.append(
                    PDFTextBlock(
                        coords=rectangle_to_coord(google_ai_block),
                        page_number=page.page_number,
                        block_id=len(document_text_blocks)+1,
                        block_type=block_type,
                        block_confidence=block_confidence,
                        block_text=layout_to_text(page.layout, page.text),
                        block_language=page.detected_languages[0].language_code
                    )
                )

        document_pages_metadata.append(
            PDFPageMetadata(
                page_number=page.page_number,
                page_width=(page_vertices.x, page_vertices.y),
            )
        )

    return PDFData(
        page_metadata=document_pages_metadata,
        text_blocks=document_text_blocks,
        md5sum=document_md5sum
    )
