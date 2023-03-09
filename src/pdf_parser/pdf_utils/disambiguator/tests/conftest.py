import pytest
from layoutparser import TextBlock, Rectangle, Layout
import random


def generate_text_block(coordinates: tuple, score: float) -> TextBlock:
    """Generates a textblock with specific coordinates and a random score."""
    return TextBlock(
        block=Rectangle(
            x_1=coordinates[0],
            y_1=coordinates[1],
            x_2=coordinates[2],
            y_2=coordinates[3],
        ),
        text=None,
        id=None,
        type="Title",
        parent=None,
        next=None,
        score=score,
    )


@pytest.fixture(scope="function")
def test_layout_random(request) -> Layout:
    """Returns a layout with the given number of text blocks."""
    return Layout(
        [
            generate_text_block(
                coordinates=(
                    random.random() * 100,
                    random.random() * 100,
                    random.random() * 100,
                    random.random() * 100,
                ),
                score=random.random(),
            )
            for i in range(request.param)
        ]
    )


@pytest.fixture
def test_layout_nested() -> Layout:
    """Returns a layout with a nested block."""

    return Layout(
        [
            generate_text_block(coordinates=(0, 0, 1000, 1000), score=random.random()),
            generate_text_block(
                coordinates=(100, 100, 500, 500), score=random.random()
            ),
        ]
    )


@pytest.fixture
def test_layout_not_nested() -> Layout:
    """Returns a layout with no nested block."""

    return Layout(
        [
            generate_text_block(coordinates=(0, 0, 1000, 1000), score=random.random()),
            generate_text_block(
                coordinates=(1100, 1100, 5000, 5000), score=random.random()
            ),
        ]
    )


@pytest.fixture
def test_layout_overlapping_vertically_nested() -> Layout:
    """
    Returns a layout with two overlapping blocks, with the inner block being vertically nested.

    I.e. the entire vertical span of the block lies within the outer block.
    """

    return Layout(
        [
            generate_text_block(coordinates=(0, 0, 1000, 1000), score=random.random()),
            generate_text_block(
                coordinates=(100, 200, 5000, 800), score=random.random()
            ),
        ]
    )


@pytest.fixture
def test_layout_overlapping_horizontally_nested() -> Layout:
    """
    Returns a layout with two overlapping blocks, with the inner block being horizontally nested.

    I.e. the entire horizontal span of the block lies within the outer block.
    """

    return Layout(
        [
            generate_text_block(coordinates=(0, 0, 1000, 1000), score=random.random()),
            generate_text_block(
                coordinates=(100, 500, 500, 5000), score=random.random()
            ),
        ]
    )


@pytest.fixture
def test_layout_overlapping_horizontal_vertically_symmetrical() -> Layout:
    """Returns a layout with two overlapping blocks, having identical vertical coordinates but differing horizontal."""

    return Layout(
        [
            generate_text_block(coordinates=(0, 0, 1000, 1000), score=random.random()),
            generate_text_block(
                coordinates=(100, 0, 5000, 1000), score=random.random()
            ),
        ]
    )


@pytest.fixture
def test_layout_two_identical_blocks() -> Layout:
    """Returns a layout with two identical blocks."""

    return Layout(
        [
            generate_text_block(coordinates=(0, 0, 1000, 1000), score=random.random()),
            generate_text_block(coordinates=(0, 0, 1000, 1000), score=random.random()),
        ]
    )


@pytest.fixture
def test_layout_overlapping_in_both_axis() -> Layout:
    """Returns a layout with two blocks overlapping in the x and y-axis."""

    return Layout(
        [
            generate_text_block(coordinates=(0, 0, 1000, 1000), score=random.random()),
            generate_text_block(
                coordinates=(500, 500, 1500, 1500), score=random.random()
            ),
        ]
    )


@pytest.fixture
def test_layout_restrictive() -> Layout:
    """Returns a layout with very low confidence scores."""

    return Layout(
        [
            generate_text_block(
                coordinates=(
                    random.random() * 100,
                    random.random() * 100,
                    random.random() * 100,
                    random.random() * 100,
                ),
                score=random.random() * 100,
            )
            for i in range(100)
        ]
    )


@pytest.fixture
def soft_margin() -> dict:
    """Returns a soft margin for the disambiguator."""
    return {
        "left": 15,
        "right": 15,
        "top": 15,
        "bottom": 15,
    }
