import pytest

from src.pdf_parser.pdf_utils.disambiguator.pipeline import run_disambiguation


@pytest.mark.parametrize("test_layout_random", [100], indirect=["test_layout_random"])
def test_run_disambiguation(test_layout_nested, test_layout_random):
    """Test that running the disambiguation pipeline gets the correct response."""
    layout = test_layout_nested + test_layout_random

    # TODO add these values from config
    disambiguated_layout = run_disambiguation(
        layout=layout,
        restrictive_model_threshold=0.4,
        unnest_soft_margin=15,
        min_overlapping_pixels_horizontal=5,
        min_overlapping_pixels_vertical=5,
    )

    assert disambiguated_layout != layout
    assert len(disambiguated_layout) < len(layout)
    # TODO add assertion that no boxes are nested
