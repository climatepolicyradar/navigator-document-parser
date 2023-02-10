import pytest
from layoutparser import Layout, TextBlock, Rectangle

from src.pdf_parser.pdf_utils.postprocess_layout import (
    infer_column_groups,
    filter_inferred_blocks,
    infer_missing_blocks_from_gaps,
)


@pytest.fixture
def layout_testcase_1():
    """Create a layout where the probable interpretation is 2 columns with some undetected text blocks."""
    # Setup a layout where the natural interpretation is 2 columns with some undetected text blocks.
    block_1 = Rectangle(x_1=0, y_1=100, x_2=20, y_2=120)
    block_2 = Rectangle(x_1=5, y_1=170, x_2=25, y_2=190)
    block_3 = Rectangle(x_1=30, y_1=110, x_2=55, y_2=130)
    block_4 = Rectangle(x_1=35, y_1=140, x_2=50, y_2=180)
    # Create text blocks.
    text_block_1 = TextBlock(block_1, text="text block 1")
    text_block_2 = TextBlock(block_2, text="text block 2")
    text_block_3 = TextBlock(block_3, text="text block 3")
    text_block_4 = TextBlock(block_4, text="text block 4")
    # Create layout.
    layout = Layout(
        [text_block_1, text_block_2, text_block_3, text_block_4], page_data=5
    )
    return layout


@pytest.fixture
def layout_testcase_2():
    # Setup a layout where the natural interpretation is 2 columns with some undetected text blocks.
    block_1 = Rectangle(x_1=0, y_1=0, x_2=10, y_2=10)
    inferred_block = Rectangle(x_1=5, y_1=7, x_2=9, y_2=16)
    block_2 = Rectangle(x_1=0, y_1=15, x_2=12, y_2=20)
    # Simple geometry shows that the inferred block has 4/9 of its area covered by both blocks.

    # Create text blocks.
    text_block_1 = TextBlock(block_1, text="text block 1")
    inferred_text_block = TextBlock(
        inferred_block, text="inferred text block", type="Inferred from gaps", score=1.0
    )
    text_block_2 = TextBlock(block_2, text="text block 2")
    # Create layout.
    layout = Layout([text_block_1, inferred_text_block, text_block_2])
    return layout


def test_infer_column_groups(layout_testcase_1):
    """Test that column groups are inferred correctly."""
    # Setup.
    column_groups = infer_column_groups(layout_testcase_1)
    # There should be 2 columns by construction.
    assert len(set(column_groups)) == 2


def test_infer_missing_blocks_from_gaps(layout_testcase_1):
    """Test that missing blocks are inferred correctly."""
    # Generic setup.
    page_height = 500
    start_pixel_buffer = 20
    end_pixel_buffer = 20
    layout_testcase_1_first_col = layout_testcase_1[0:2]
    missing_blocks = infer_missing_blocks_from_gaps(
        layout_testcase_1_first_col,
        page_height=page_height,
        start_pixel_buffer=start_pixel_buffer,
        end_pixel_buffer=end_pixel_buffer,
    )
    # Setup case 1. There is a missing block between block 1 and block 2 in the first column.
    x_1, y_1, x_2, y_2 = (0, 120, 25, 170)
    expected_missing_rectangle = Rectangle(x_1=x_1, y_1=y_1, x_2=x_2, y_2=y_2)
    expected_inferred_block = TextBlock(
        expected_missing_rectangle, type="Inferred from gaps", score=1.0
    )
    assert expected_inferred_block in missing_blocks
    # Setup case 2. There is a missing block at the start of a page in the first column.
    x_1, y_1, x_2, y_2 = (0, start_pixel_buffer, 25, 100)
    expected_missing_rectangle = Rectangle(x_1=x_1, y_1=y_1, x_2=x_2, y_2=y_2)
    expected_inferred_block = TextBlock(
        expected_missing_rectangle, type="Inferred from gaps", score=1.0
    )
    assert expected_inferred_block in missing_blocks
    # Setup case 3. There is a missing block at the end of a page in the first column.
    x_1, y_1, x_2, y_2 = (0, 190, 25, page_height - end_pixel_buffer)
    expected_missing_rectangle = Rectangle(x_1=x_1, y_1=y_1, x_2=x_2, y_2=y_2)
    expected_inferred_block = TextBlock(
        expected_missing_rectangle, type="Inferred from gaps", score=1.0
    )
    assert expected_inferred_block in missing_blocks


def test_filter_inferred_blocks(layout_testcase_2):
    # simple geometry shows that the inferred block has 4/9 of its area covered by both blocks for testcase 2.
    # Case 1: Coverage exceeds threshold.
    filtered_layout = filter_inferred_blocks(layout_testcase_2, remove_threshold=5 / 9)
    assert len(filtered_layout) == 3
    # Case 2: Coverage does not exceed threshold.
    filtered_layout = filter_inferred_blocks(layout_testcase_2, remove_threshold=1 / 3)
    assert len(filtered_layout) == 2
