import pytest
import layoutparser as lp
from pathlib import Path

from src.pdf_parser.pdf_utils.parsing_utils import PostProcessor, LayoutDisambiguator


@pytest.fixture
def test_page():
    """Load a page with useful test properties."""
    pdf_path = (
        Path(__file__).parent
        / "data"
        / "BRB-2019-12-25-National Energy Policy 2019-2030_19fbfbb2c35d8f43bfa1b8c3219605b4.pdf"
    )
    _, pdf_images = lp.load_pdf(pdf_path, load_images=True)
    pdf_image = pdf_images[47]
    return pdf_image


@pytest.fixture
def base_model():
    return lp.Detectron2LayoutModel(
        config_path="lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x",  # In model catalog,
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
        device="cpu",
    )


@pytest.fixture
def disambiguator(test_page, base_model):
    disambiguator = LayoutDisambiguator(test_page, base_model)
    return disambiguator


@pytest.fixture
def disambiguated_layout(disambiguator):
    return disambiguator.disambiguate_layout()


@pytest.fixture
def postprocessor(disambiguated_layout):
    postprocessor = PostProcessor(disambiguated_layout)
    return postprocessor


@pytest.fixture
def layout_testcase_1():
    """Create a layout where the probable interpretation is 2 columns with some undetected text blocks."""
    # Setup a layout where the natural interpretation is 2 columns with some undetected text blocks.
    block_1 = lp.Rectangle(x_1=0, y_1=0, x_2=20, y_2=20)
    block_2 = lp.Rectangle(x_1=5, y_1=70, x_2=25, y_2=90)
    block_3 = lp.Rectangle(x_1=30, y_1=10, x_2=55, y_2=30)
    block_4 = lp.Rectangle(x_1=35, y_1=40, x_2=50, y_2=80)
    # Create text blocks.
    text_block_1 = lp.TextBlock(block_1, text="text block 1")
    text_block_2 = lp.TextBlock(block_2, text="text block 2")
    text_block_3 = lp.TextBlock(block_3, text="text block 3")
    text_block_4 = lp.TextBlock(block_4, text="text block 4")
    # Create layout.
    layout = lp.Layout([text_block_1, text_block_2, text_block_3, text_block_4])
    return layout


@pytest.fixture
def layout_testcase_2():
    # Setup a layout where the natural interpretation is 2 columns with some undetected text blocks.
    block_1 = lp.Rectangle(x_1=0, y_1=0, x_2=10, y_2=10)
    inferred_block = lp.Rectangle(x_1=5, y_1=7, x_2=9, y_2=16)
    block_2 = lp.Rectangle(x_1=0, y_1=15, x_2=12, y_2=20)
    # Simple geometry shows that the inferred block has 4/9 of its area covered by both blocks.

    # Create text blocks.
    text_block_1 = lp.TextBlock(block_1, text="text block 1")
    inferred_text_block = lp.TextBlock(
        inferred_block, text="inferred text block", type="Inferred from gaps", score=1.0
    )
    text_block_2 = lp.TextBlock(block_2, text="text block 2")
    # Create layout.
    layout = lp.Layout([text_block_1, inferred_text_block, text_block_2])
    return layout


def test_unnest_boxes(disambiguator):
    """Test that unnest boxes works as expected."""
    # Setup.
    pixel_margin = 15
    soft_margin = {
        "top": pixel_margin,
        "bottom": pixel_margin,
        "left": pixel_margin,
        "right": pixel_margin,
    }
    unnested_layout = disambiguator._unnest_boxes(pixel_margin)
    # Make sure no box is within another box (soft margin).
    for box in unnested_layout:
        for other_box in unnested_layout:
            if box != other_box:
                assert not box.is_in(other_box, soft_margin)


def test_infer_column_groups(layout_testcase_1, postprocessor):
    """Test that column groups are inferred correctly."""
    # Setup.
    column_groups = postprocessor._infer_column_groups(layout_testcase_1)
    # There should be 2 columns by construction.
    assert len(set(column_groups)) == 2


def test_infer_missing_blocks_from_gaps(layout_testcase_1, postprocessor):
    """Test that missing blocks are inferred correctly."""
    # Setup.
    x_1, y_1, x_2, y_2 = (0, 20, 25, 70)
    expected_missing_rectangle = lp.Rectangle(x_1=x_1, y_1=y_1, x_2=x_2, y_2=y_2)
    expected_inferred_block = lp.TextBlock(
        expected_missing_rectangle, type="Inferred from gaps", score=1.0
    )
    layout_testcase_1_first_col = layout_testcase_1[0:2]
    missing_block = postprocessor._infer_missing_blocks_from_gaps(
        layout_testcase_1_first_col
    )[0]
    assert missing_block == expected_inferred_block


def test_filter_inferred_blocks(layout_testcase_2, postprocessor):
    # simple geometry shows that the inferred block has 4/9 of its area covered by both blocks for testcase 2.
    # Case 1: Coverage exceeds threshold.
    filtered_layout = postprocessor._filter_inferred_blocks(
        layout_testcase_2, remove_threshold=5 / 9
    )
    assert len(filtered_layout) == 3
    # Case 2: Coverage does not exceed threshold.
    filtered_layout = postprocessor._filter_inferred_blocks(
        layout_testcase_2, remove_threshold=1 / 3
    )
    assert len(filtered_layout) == 2
