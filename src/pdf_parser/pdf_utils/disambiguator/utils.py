from layoutparser import Layout, TextBlock
from shapely.geometry import Polygon
from shapely.ops import unary_union
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
    restrictive_boxes = []
    permissive_boxes = []

    for box in layout:
        if box.score > restrictive_model_threshold:
            restrictive_boxes.append(box)
        else:
            permissive_boxes.append(box)

    return Layout(restrictive_boxes), Layout(permissive_boxes)


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
    boxes_above_combination_threshold = [
        box
        for indx, box in enumerate(layout_permissive)
        if unexplained_fractions[indx] > combination_threshold
    ]
    for box in boxes_above_combination_threshold:
        box.type = "Ambiguous"

    layout_combined = layout_restrictive + Layout(boxes_above_combination_threshold)
    return layout_combined


def lp_coords_to_shapely_polygon(coords: Tuple[float, float, float, float]) -> Polygon:
    """Convert layoutparser coordinates to shapely format so that we can use convenient shapely ops.

    The coord format is as follows:

    [(x_top_left, y_top_left, x_bottom_right, y_bottom_right)] - > [(x_bottom_left, y_bottom_left), (x_top_left,
    y_top_left), (x_top_right, y_top_right), (x_bottom_right, y_bottom_right)]

    Args:
        coords: The layoutparser coordinates for the box.

    Returns:
        The shapely polygon.
    """
    shapely_coords = [
        (coords[0], coords[1]),
        (coords[0], coords[3]),
        (coords[2], coords[3]),
        (coords[2], coords[1]),
    ]
    return Polygon(shapely_coords)


def calculate_unexplained_fractions(
    restrictive_layout: Layout, permissive_layout: Layout
) -> tuple[Layout, list[float]]:
    """Calculate the fraction of each box in the permissive layout not captured by boxes in the restrictive layout.

    This is useful because we want to find boxes that are not already accounted for by the strict model but that may
    contain useful text.

    Args:
        restrictive_layout: The layout with boxes above the restrictive threshold.
        permissive_layout: The layout with boxes below the restrictive threshold.

    Returns:
        A LayoutWithFractions object containing a layout model.
    """

    # Get the polygons for each box in the strict and unfiltered layouts.
    restrictive_polygons = [
        lp_coords_to_shapely_polygon(box.coordinates) for box in restrictive_layout
    ]
    permissive_polygons = [
        lp_coords_to_shapely_polygon(box.coordinates) for box in permissive_layout
    ]
    unexplained_fractions = []
    for poly in permissive_polygons:
        poly_unexplained = poly.difference(unary_union(restrictive_polygons))
        area_unexplained = poly_unexplained.area
        area_total = poly.area
        frac_unexplained = area_unexplained / area_total
        unexplained_fractions.append(frac_unexplained)

    return permissive_layout, unexplained_fractions
