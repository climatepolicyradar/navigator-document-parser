import pytest

from src.pdf_parser.pdf_utils.disambiguator.unexplained import (
    calculate_unexplained_fractions,
)


@pytest.mark.parametrize("test_layout_random", [100], indirect=["test_layout_random"])
def test_calculate_unexplained_fractions(test_layout_random, test_layout_restrictive):
    """Tests that the unexplained fraction is calculated correctly."""
    permissive_layout, unexplained_fractions = calculate_unexplained_fractions(
        restrictive_layout=test_layout_restrictive, permissive_layout=test_layout_random
    )

    assert all([0 <= f <= 1 for f in unexplained_fractions])

    assert len(unexplained_fractions) == len(permissive_layout)
