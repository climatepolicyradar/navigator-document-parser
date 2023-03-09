import itertools
from typing import Optional, Tuple

from layoutparser import Layout, TextBlock, Rectangle

from src.pdf_parser.pdf_utils.disambiguator.utils import is_in


def horizontal_overlap(box_1: TextBlock, box_2: TextBlock) -> bool:
    """Check if the boxes overlap horizontally."""
    if (
        box_1.coordinates[0] < box_2.coordinates[2]
        and box_1.coordinates[2] > box_2.coordinates[0]
    ):
        return True
    return False


def vertical_overlap(box_1: TextBlock, box_2: TextBlock) -> bool:
    """Check if the boxes overlap vertically."""
    if (
        box_1.coordinates[1] < box_2.coordinates[3]
        and box_1.coordinates[3] > box_2.coordinates[1]
    ):
        return True
    return False


def reduce_overlapping_boxes_vertical(
    top_box: TextBlock,
    bottom_box: TextBlock,
    min_pixel_overlap_vertical: Optional[int] = 5,
) -> Tuple[TextBlock, TextBlock]:
    """Reduce the size of overlapping boxes to eliminate overlaps in the vertical axis."""
    intersection_height = top_box.intersect(bottom_box).height
    if intersection_height > min_pixel_overlap_vertical:
        if horizontally_nested(top_box, bottom_box):
            top_box = TextBlock(
                block=Rectangle(
                    x_1=top_box.coordinates[0],
                    y_1=top_box.coordinates[1] + intersection_height,
                    x_2=top_box.coordinates[2],
                    y_2=top_box.coordinates[3],
                ),
                text=top_box.text,
                id=top_box.id,
                type=top_box.type,
                parent=top_box.parent,
                next=top_box.next,
                score=top_box.score,
            )

        elif horizontally_nested(bottom_box, top_box):
            bottom_box = TextBlock(
                block=Rectangle(
                    x_1=bottom_box.coordinates[0],
                    y_1=bottom_box.coordinates[1],
                    x_2=bottom_box.coordinates[2],
                    y_2=bottom_box.coordinates[3] - intersection_height,
                ),
                text=bottom_box.text,
                id=bottom_box.id,
                type=bottom_box.type,
                parent=bottom_box.parent,
                next=bottom_box.next,
                score=bottom_box.score,
            )
        return top_box, bottom_box
    return top_box, bottom_box


def reduce_overlapping_boxes_horizontal(
    left_box: TextBlock,
    right_box: TextBlock,
    min_pixel_overlap_horizontal: Optional[int] = 5,
) -> Tuple[TextBlock, TextBlock]:
    """Reduce the size of overlapping boxes to eliminate overlaps in the horizontal axis."""
    intersection_width = left_box.intersect(right_box).width
    if intersection_width > min_pixel_overlap_horizontal:
        if vertically_nested(left_box, right_box):
            left_box = TextBlock(
                block=Rectangle(
                    x_1=left_box.coordinates[0],
                    y_1=left_box.coordinates[1],
                    x_2=left_box.coordinates[2] - intersection_width,
                    y_2=left_box.coordinates[3],
                ),
                text=left_box.text,
                id=left_box.id,
                type=left_box.type,
                parent=left_box.parent,
                next=left_box.next,
                score=left_box.score,
            )
        elif vertically_nested(right_box, left_box):
            right_box = TextBlock(
                block=Rectangle(
                    x_1=right_box.coordinates[0] + intersection_width,
                    y_1=right_box.coordinates[1],
                    x_2=right_box.coordinates[2],
                    y_2=right_box.coordinates[3],
                ),
                text=right_box.text,
                id=right_box.id,
                type=right_box.type,
                parent=right_box.parent,
                next=right_box.next,
                score=right_box.score,
            )
        return left_box, right_box
    return left_box, right_box


def is_on_top(box1: TextBlock, box2: TextBlock) -> bool:
    """Identify if box1 is on top of the bottom box."""
    if box1.coordinates[1] > box2.coordinates[1]:
        return True
    return False


def is_on_left(box1: TextBlock, box2: TextBlock) -> bool:
    """Identify if box1 is on the left of the right box."""
    if box1.coordinates[0] < box2.coordinates[0]:
        return True
    return False


def vertically_nested(box1: TextBlock, box2: TextBlock) -> bool:
    """
    Identify if box1 is vertically nested in box2 or if box2 is vertically nested in box1.

    I.e. (0,1,2,3) is vertically nested within (1,0,2,5) as the entire range of y values lie within the other box.
    """
    if (
        box1.coordinates[1] > box2.coordinates[1]
        and box1.coordinates[3] < box2.coordinates[3]
        or box2.coordinates[1] > box1.coordinates[1]
        and box2.coordinates[3] < box1.coordinates[3]
    ):
        return True
    return False


def horizontally_nested(box1: TextBlock, box2: TextBlock) -> bool:
    """
    Identify if box1 is horizontally nested in box2 or if box2 is horizontally nested in box1.

    I.e. (2,0,4,100) is nested horizontally within (0,0,10,10) as the entire range of x values lie within the other box.
    """
    if (
        box1.coordinates[0] > box2.coordinates[0]
        and box1.coordinates[2] < box2.coordinates[2]
        or box2.coordinates[0] > box1.coordinates[0]
        and box2.coordinates[2] < box1.coordinates[2]
    ):
        return True
    return False


def overlap_in_both_axis(box1: TextBlock, box2: TextBlock) -> bool:
    """
    Identify if the boxes overlap in both axis.

    I.e. (0,0,10,10) and (5,5,15,15) overlap in both axis.
    """
    if (horizontal_overlap(box1, box2) and not horizontally_nested(box1, box2)) and (
        vertical_overlap(box1, box2) and not vertically_nested(box1, box2)
    ):
        return True
    return False


def reduce_overlapping_boxes(
    layout: Layout,
    unnest_soft_margin: int = 5,
    min_pixel_overlap_vertical: int = 5,
    min_pixel_overlap_horizontal: int = 5,
) -> Layout:
    """
    Eliminate where possible overlapping boxes by reducing their size by the minimal amount necessary.

    In general, for every pair of rectangular boxes with coordinates of
    the form (x_top_left, y_top_left, x_bottom_right, y_bottom_right),
    we want to reshape them to eliminate the intersecting regions in the
    alignment direction. For example, if we want to eliminate overlaps of
    the following two rectangles the transformation should be:

    [(0,0,3,3),(1,1,2,4)] -> [(0,0,3,3), (1,3,2,4)]

    Args:
        layout: The layout to reduce.
        unnest_soft_margin: The soft margin to use when unnesting boxes.
        min_pixel_overlap_vertical: The minimal pixel overlap needed to reduce boxes in vertical direction.
        min_pixel_overlap_horizontal: The minimal pixel overlap needed to reduce boxes in horizontal direction.

    Returns:
        The new layout with blocks having no overlapping coordinates.
    """
    soft_margin = {
        "top": unnest_soft_margin,
        "bottom": unnest_soft_margin,
        "left": unnest_soft_margin,
        "right": unnest_soft_margin,
    }

    for (indx1, box1), (indx2, box2) in itertools.combinations(enumerate(layout), 2):
        if (
            box1.intersect(box2).area > 0
            and box1 != box2
            and not is_in(box1, box2, soft_margin)
            and not is_in(box2, box1, soft_margin)
            and not overlap_in_both_axis(box1, box2)
        ):
            if vertical_overlap(box1, box2) and not vertically_nested(box1, box2):
                top_box, bottom_box = reduce_overlapping_boxes_vertical(
                    top_box=(box1 if is_on_top(box1, box2) else box2),
                    bottom_box=(box2 if is_on_top(box1, box2) else box1),
                    min_pixel_overlap_vertical=min_pixel_overlap_vertical,
                )
                box1 = top_box if is_on_top(box1, box2) else bottom_box
                box2 = bottom_box if is_on_top(box1, box2) else top_box

            if horizontal_overlap(box1, box2) and not horizontally_nested(box1, box2):
                left_box, right_box = reduce_overlapping_boxes_horizontal(
                    left_box=(box1 if is_on_left(box1, box2) else box2),
                    right_box=(box2 if is_on_left(box1, box2) else box1),
                    min_pixel_overlap_horizontal=min_pixel_overlap_horizontal,
                )
                box1 = left_box if is_on_left(box1, box2) else right_box
                box2 = right_box if is_on_left(box1, box2) else left_box

            layout[indx1] = box1
            layout[indx2] = box2

    return layout
