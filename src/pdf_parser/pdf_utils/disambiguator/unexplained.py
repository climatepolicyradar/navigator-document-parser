import logging

from layoutparser import Layout
from shapely.geometry import Polygon
from shapely.ops import unary_union
from typing import Tuple

_LOGGER = logging.getLogger(__name__)


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
