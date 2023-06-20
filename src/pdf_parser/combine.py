import google
from layout_parser import Rectangle

from src.base import ParserOutput
from src.pdf_parser.layout import LayoutParserWrapper


def assign_block_type(document: google.cloud.documentai_v1.Document, lp_obj: LayoutParserWrapper) -> ParserOutput:
    """The google document ai api has many good features, however it does not support text block type detection.

    For example ‘Table’ or ‘Figure’.
    This is necessary as we are may want to filter these out at a later date etc.

    To solve this problem we want to use layout parser to detect types and boxes in the documents and assign the
    types detected in layout parser to text blocks identified from the google ai api.
    """
    for page in document.pages:
        layout = lp_obj.get_layout(page.image.content)
        layout_coords = [block.block for block in layout._blocks]

        page_vertices = page.layout.bounding_poly.vertices[2]

        google_ai_blocks = [block.layout.bounding_poly.normalized_vertices for block in page.blocks]
        google_ai_coords = [
            Rectangle(x_1=block[0].x, y_1=block[0].y, x_2=block[2].x, y_2=block[2].y) for block in google_ai_blocks
        ]

        google_ai_coords_scaled = [
            Rectangle(
                x_1=block.x_ 1*page_vertices.x,
                y_1=block.y_ 1*page_vertices.y,
                x_2=block.x_ 2*page_vertices.x,
                y_2=block.y_ 2*page_vertices.y
            )
            for block in google_ai_coords
        ]

        for layout_block in layout_coords:
            for google_ai_block in google_ai_coords_scaled:
                if layout_block.intersect(google_ai_block).area / layout_block.area > 0.7:
                    print('overlap')
                    print(layout_block)
                    print(google_ai_block)
                    print('---')
                    # FIXME: Add code to assign block type to google ai block

        # FIXME: THis is a temporary fix to get the code to look right
        return ParserOutput(document=document, layout=layout)