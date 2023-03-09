import pytest

from src.pdf_parser.pdf_utils.disambiguator.unexplained import (
    calculate_unexplained_fractions,
)
from src.pdf_parser.pdf_utils.disambiguator.utils import (
    is_in,
    split_layout,
    combine_layouts,
)


def test_is_in(test_layout_nested, soft_margin):
    """Tests that the is_in function works correctly."""
    # TODO could do this dynamically by comparing coordinates
    nested_box = test_layout_nested[1]
    not_nested_box = test_layout_nested[0]

    assert is_in(nested_box, not_nested_box, soft_margin=soft_margin)
    assert not is_in(not_nested_box, nested_box, soft_margin=soft_margin)


@pytest.mark.parametrize("test_layout_random", [100], indirect=["test_layout_random"])
def test_split(test_layout_random):
    """Tests that the split function works correctly."""
    layout_restrictive, layout_permissive = split_layout(
        test_layout_random, restrictive_model_threshold=0.4
    )

    assert len(layout_restrictive) + len(layout_permissive) == len(test_layout_random)
    for box in layout_restrictive:
        assert box.score >= 0.4

    for box in layout_permissive:
        assert box.score < 0.4


@pytest.mark.parametrize("test_layout_random", [100], indirect=["test_layout_random"])
def test_combine(test_layout_restrictive, test_layout_random):
    """Tests that the combine function works correctly."""
    permissive_layout, unexplained_fractions = calculate_unexplained_fractions(
        restrictive_layout=test_layout_restrictive, permissive_layout=test_layout_random
    )

    combined_layout = combine_layouts(
        layout_restrictive=test_layout_restrictive,
        layout_permissive=permissive_layout,
        unexplained_fractions=unexplained_fractions,
        combination_threshold=0.5,
    )
    assert len(combined_layout) >= len(test_layout_restrictive)
    assert len(permissive_layout) + len(
        [i for i in unexplained_fractions if i > 0.5]
    ) == len(combined_layout)
