from src.pdf_parser.pdf_utils.disambiguator.nested import remove_nested_boxes


def test_remove_nested_boxes(test_layout_nested, test_layout_not_nested):
    """Tests that nested blocks are removed."""

    layout = remove_nested_boxes(test_layout_nested)
    assert len(layout) == 1

    layout = remove_nested_boxes(test_layout_not_nested)
    assert len(layout) == 2


# def test_removed_nested_boxes_time(test_layout_random):
#     """
#     Tests that nested blocks are removed in a reasonable amount of time.
#     """
#     # TODO how to time that this doesn't take too long, could input small lengths and assert that relationship
#     #  between layout length and time is not exponential?
#     # TODO layout = remove_nested_boxes(test_layout_random(length=150))
#     pass
