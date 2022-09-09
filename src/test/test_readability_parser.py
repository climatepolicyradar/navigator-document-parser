from typing import Tuple

import pytest

from src.readability import ReadabilityParser


@pytest.mark.parametrize(
    "list_headers",
    [
        ("1.", "2."),
        ("(iv)", "(v)"),
        ("IX.", "X."),
        ("a.", "b."),
        ("(c)", "(d)"),
        ("i.", "ii."),
        ("•", "•"),
        ("-", "-"),
        ("–", "–"),
        ("*", "*"),
    ],
)
def test_combining_bullets(list_headers: Tuple[str, str]) -> None:
    """Test the ReadabilityParser method to combine bullet points with the next line."""

    parser = ReadabilityParser()
    text_by_line = [
        list_headers[0],
        "This is a bullet point",
        list_headers[1],
        "This is another bullet point",
        "This is a normal line",
    ]
    combined = parser._combine_bullet_lines_with_next(text_by_line)

    assert combined == [
        f"{list_headers[0]} This is a bullet point",
        f"{list_headers[1]} This is another bullet point",
        "This is a normal line",
    ]
