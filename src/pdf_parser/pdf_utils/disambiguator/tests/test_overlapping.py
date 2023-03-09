from src.pdf_parser.pdf_utils.disambiguator.overlapping import reduce_overlapping_boxes


def test_reduce_overlapping_boxes(
    test_layout_two_identical_blocks,
    test_layout_overlapping_horizontally_nested,
    test_layout_overlapping_vertically_nested,
    test_layout_overlapping_horizontal_vertically_symmetrical,
    test_layout_overlapping_in_both_axis,
):
    """Test the reduce_overlapping_boxes function with different scenarios."""

    # Identical blocks are technically nested so we expect no reduction to occur
    reduced_layout = reduce_overlapping_boxes(
        layout=test_layout_two_identical_blocks,
        unnest_soft_margin=5,
        min_pixel_overlap_vertical=5,
        min_pixel_overlap_horizontal=5,
    )
    assert reduced_layout[0].coordinates == (0, 0, 1000, 1000)
    assert reduced_layout[1].coordinates == (0, 0, 1000, 1000)
    assert (
        reduced_layout[0].block.intersect(reduced_layout[1].block).area
        == reduced_layout[0].block.area
    )

    # Overlapping layouts should be reduced.
    reduced_layout = reduce_overlapping_boxes(
        layout=test_layout_overlapping_horizontally_nested,
        unnest_soft_margin=5,
        min_pixel_overlap_vertical=5,
        min_pixel_overlap_horizontal=5,
    )
    assert reduced_layout[0].coordinates == (0, 0, 1000, 1000)
    assert reduced_layout[1].coordinates == (100, 1000, 500, 5000)
    assert reduced_layout[0].block.intersect(reduced_layout[1].block).area == 0

    # Overlapping layouts should be reduced.
    reduced_layout = reduce_overlapping_boxes(
        layout=test_layout_overlapping_vertically_nested,
        unnest_soft_margin=5,
        min_pixel_overlap_vertical=5,
        min_pixel_overlap_horizontal=5,
    )
    assert reduced_layout[0].coordinates == (0, 0, 100, 1000)
    assert reduced_layout[1].coordinates == (100, 200, 5000, 800)
    assert reduced_layout[0].block.intersect(reduced_layout[1].block).area == 0

    # TODO check this. Is this the correct behaviour?
    # Boxes that are symmetrical in one axis are considered nested thus we expect no reduction to occur
    reduced_layout = reduce_overlapping_boxes(
        layout=test_layout_overlapping_horizontal_vertically_symmetrical,
        unnest_soft_margin=5,
        min_pixel_overlap_vertical=5,
        min_pixel_overlap_horizontal=5,
    )
    assert reduced_layout[0].coordinates == (0, 0, 1000, 1000)
    assert reduced_layout[1].coordinates == (100, 0, 5000, 1000)
    assert reduced_layout[0].block.intersect(reduced_layout[1].block).area > 0

    # Boxes that overlap in two axis can't be reduced without missing an area so we expect no reduction to occur
    reduced_layout = reduce_overlapping_boxes(
        layout=test_layout_overlapping_in_both_axis,
        unnest_soft_margin=5,
        min_pixel_overlap_vertical=5,
        min_pixel_overlap_horizontal=5,
    )
    assert reduced_layout[0].coordinates == (0, 0, 1000, 1000)
    assert reduced_layout[1].coordinates == (500, 500, 1500, 1500)
    assert reduced_layout[0].block.intersect(reduced_layout[1].block).area > 0
