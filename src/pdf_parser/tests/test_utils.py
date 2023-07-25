from azure.ai.formrecognizer import Point, DocumentParagraph, DocumentTable

from src.base import PDFTextBlock, PDFTableBlock
from src.pdf_parser.utils import (
    polygon_to_coords,
    convert_to_text_block,
    convert_to_table_block
)


def test_valid_polygon_to_coords() -> None:
    """Test that we can convert a sequence of points into a list of coordinates."""
    valid_points = [
        Point(x=0.0, y=1.0), Point(x=1.0, y=1.0),
        Point(x=1.0, y=0.0), Point(x=0.0, y=0.0)
    ]

    coords = polygon_to_coords(valid_points)
    assert isinstance(coords, list)
    for coord in coords:
        assert isinstance(coord, tuple)
        for coord_val in coord:
            assert isinstance(coord_val, float)


def test_invalid_polygon_to_coords() -> None:
    """Test that we throw an exception should the polygon not be of the correct form."""

    invalid_points = [Point(x=0.0, y=1.0,), Point(x=1.0, y=1.0)]

    coords = None
    error = None
    try:
        coords = polygon_to_coords(invalid_points)
    except ValueError as e:
        error = e

    assert error.__class__ is ValueError
    assert coords is None


def test_convert_to_text_block(document_paragraph: DocumentParagraph) -> None:
    """Test that we can convert an Azure document paragraph to a text block."""
    text_block = convert_to_text_block(paragraph_id=1, paragraph=document_paragraph)

    # Pydantic will validate the types so not alot more validation needed
    assert isinstance(text_block, PDFTextBlock)
    assert text_block.type == "Document Header"


def test_convert_to_table_block(document_table: DocumentTable) -> None:
    """Test that we can successfully assign data from a document table to a pdf table
    block."""
    index = 123
    table_block = convert_to_table_block(document_table, index=index)

    assert isinstance(table_block, PDFTableBlock)
    assert table_block.table_id == str(index)
    assert table_block.row_count is document_table.row_count
    assert table_block.column_count is document_table.column_count
    assert len(table_block.cells) is len(document_table.cells)


def test_convert_to_parser_output():
    pass


def test_propagate_with_correct_page():
    pass


def test_merge_responses():
    pass


def test_split_into_pages():
    pass
