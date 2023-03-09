from layoutparser import Layout, TextBlock
from typing import Tuple


def is_in(nested_box: TextBlock, box: TextBlock, soft_margin: dict) -> bool:
    """
    Identify if the nested box is in the main box.

    Args:
        nested_box: The box that may be contained within the main box.
        box: The main box.
        soft_margin: Allowing lee-way for the nested box to be contained within the main box.
    """
    if (
        nested_box.block.x_1 > box.block.x_1 - soft_margin["left"]
        and nested_box.block.x_2 < box.block.x_2 + soft_margin["right"]
        and nested_box.block.y_1 > box.block.y_1 - soft_margin["top"]
        and nested_box.block.y_2 < box.block.y_2 + soft_margin["bottom"]
    ):
        return True
    return False


def split_layout(
    layout: Layout, restrictive_model_threshold: float = 0.5
) -> Tuple[Layout, Layout]:
    """Split layout into boxes above and below a given model confidence score.

    Args:
        layout: The layout to create polygons from.
        restrictive_model_threshold: The threshold above which to include boxes in the restrictive layout.

    Returns:
        A tuple of layouts, the first with boxes above the threshold and the second with boxes below the threshold.
    """
    layout_restrictive = Layout(
        [box for box in layout if box.score > restrictive_model_threshold]
    )
    layout_permissive = Layout(
        [box for box in layout if box.score <= restrictive_model_threshold]
    )
    return layout_restrictive, layout_permissive


def combine_layouts(
    layout_restrictive: Layout,
    layout_permissive: Layout,
    unexplained_fractions: list[float],
    combination_threshold: float,
) -> Layout:
    """Add unexplained text boxes to the strict layout to get a combined layout.

    Args:
        layout_restrictive: The layout with boxes above the restrictive threshold.
        layout_permissive: The layout with boxes below the restrictive threshold.
        unexplained_fractions: The fraction of each box in the permissive layout not captured by boxes in the restrictive.
        combination_threshold: The threshold above which to include boxes from the permissive layout.

    Returns:
        The layout with boxes from the unfiltered perspective added if their areas aren't already sufficiently accounted for..
    """
    boxes_to_add = []
    for ix, box in enumerate(layout_permissive):
        # If the box's area is not "explained away" by the strict layout, add it to the combined layout with an
        # ambiguous type tag. We can use heuristics to determine its type downstream.
        if unexplained_fractions[ix] > combination_threshold:
            box.type = "Ambiguous"
            boxes_to_add.append(box)
    layout_combined = layout_restrictive + Layout(boxes_to_add)
    return layout_combined
