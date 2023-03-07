from layoutparser import Layout, TextBlock, Rectangle, Detectron2LayoutModel  # type: ignore
from PIL.PpmImagePlugin import PpmImageFile
from shapely.geometry import Polygon
from shapely.ops import unary_union
from typing import List, Tuple, Optional, Any
import logging

from pydantic import Field
from src.pdf_parser.pdf_utils.utils import BaseModel

_LOGGER = logging.getLogger(__name__)


# TODO: I added this because I want to enforce that the unexplained fractions are in the same order as the boxes in
#  the layout without adding it as page metadata, as this would require formally writing checks
#   with assert. Is there a better way to do this or is this ok?
class LayoutWithFractions(BaseModel):
    """Layout with unexplained fractions added."""

    layout: Layout
    # unexplained fractions must be a list of floats between 0 and 1
    unexplained_fractions: List[float] = Field(..., ge=0, le=1)


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
) -> LayoutWithFractions:
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
    permissive_layout_with_fractions = LayoutWithFractions(
        layout=permissive_layout, unexplained_fractions=unexplained_fractions
    )
    return permissive_layout_with_fractions


def combine_layouts(
    layout_restrictive: Layout,
    layout_permissive: LayoutWithFractions,
    combination_threshold: float,
) -> Layout:
    """Add unexplained text boxes to the strict layout to get a combined layout.

    Args:
        layout_restrictive: The layout with boxes above the restrictive threshold.
        layout_permissive: The layout with boxes below the restrictive threshold.
        combination_threshold: The threshold above which to include boxes from the permissive layout.

    Returns:
        The layout with boxes from the unfiltered perspective added if their areas aren't already sufficiently accounted for..
    """
    boxes_to_add = []
    permissive_layout = layout_permissive.layout
    for ix, box in enumerate(permissive_layout):
        unexplained_fractions = layout_permissive.unexplained_fractions
        # If the box's area is not "explained away" by the strict layout, add it to the combined layout with an
        # ambiguous type tag. We can use heuristics to determine its type downstream.
        if unexplained_fractions[ix] > combination_threshold:
            box.type = "Ambiguous"
            boxes_to_add.append(box)
    layout_combined = layout_restrictive + Layout(boxes_to_add)
    return layout_combined


def reduce_overlapping_boxes(
    box_1: TextBlock,
    box_2: TextBlock,
    direction: str = "vertical",
    min_overlapping_pixels_vertical: Optional[int] = 5,
    min_overlapping_pixels_horizontal: Optional[int] = 5,
) -> Tuple[TextBlock, TextBlock]:
    """Reduce the size of overlapping boxes to elimate overlaps.

    If two boxes overlap in a given direction (vertical or horizontal), reduce the size of both in that
     direction by the minimal amount necessary to elimate overlaps.

    Args:
        box_1: The first box to compare. This box should be the upper/left box.
        box_2: The second box to compare. This box should be the lower/right box.
        direction: The direction to reduce the boxes in.
        min_overlapping_pixels_vertical: The minimal pixel overlap needed to reduce boxes in vertical direction.
        min_overlapping_pixels_horizontal: The minimal pixel overlap needed to reduce boxes in horizontal direction.

    Returns:
        The boxes with overlaps eliminated.
    """

    if direction not in ("horizontal", "vertical"):
        raise ValueError("Direction must be 'horizontal' or 'vertical'.")

    if direction == "vertical":
        assert (
            box_1.coordinates[3] < box_2.coordinates[3]
        ), "box_1 should be the upper box."
        intersection_height = box_1.intersect(box_2).height
        if intersection_height > min_overlapping_pixels_vertical:
            rect_1 = Rectangle(
                x_1=box_1.coordinates[0],
                y_1=box_1.coordinates[1],
                x_2=box_1.coordinates[2],
                y_2=box_1.coordinates[3] - intersection_height,
            )
            rect_2 = Rectangle(
                x_1=box_2.coordinates[0],
                y_1=box_2.coordinates[1] + intersection_height,
                x_2=box_2.coordinates[2],
                y_2=box_2.coordinates[3],
            )
        else:
            rect_1 = box_1
            rect_2 = box_2
    elif direction == "horizontal":
        assert (
            box_1.coordinates[0] < box_2.coordinates[0]
        ), "box_1 should be the left box."
        intersection_width = box_1.intersect(box_2).width
        if intersection_width > min_overlapping_pixels_horizontal:
            rect_1 = Rectangle(
                x_1=box_1.coordinates[0],
                y_1=box_1.coordinates[1],
                x_2=box_1.coordinates[2] - intersection_width,
                y_2=box_1.coordinates[3],
            )
            rect_2 = Rectangle(
                x_1=box_2.coordinates[0] + intersection_width,
                y_1=box_2.coordinates[1],
                x_2=box_2.coordinates[2],
                y_2=box_2.coordinates[3],
            )
        else:
            rect_1 = box_1
            rect_2 = box_2
    return rect_1, rect_2


def check_line_contained(line_1: tuple, line_2: tuple) -> bool:
    """Check if either line is fully contained in the other.

    For example, line 1 might be from 0 to 10 and line 2 might be from 5 to 7.
    Line 2 is contained in line 1 in this case. If, however, line 2 was from 15 to 17,
    it would not be contained in line 1.

    Args:
        line_1: The first line to compare. The first element should be the leftmost or bottommost point.
        line_2: The second line to compare. The first element should be the leftmost or bottommost point.

    Returns:
        True if either line is contained in the other, False otherwise.
    """
    line_1_start, line_1_end = line_1[0], line_1[1]
    line_2_start, line_2_end = line_2[0], line_2[1]
    return (line_1_start >= line_2_start and line_1_end <= line_2_end) or (
        line_2_start >= line_1_start and line_2_end <= line_1_end
    )


def check_horizontal_or_vertical_overlap(box_1: TextBlock, box_2: TextBlock) -> bool:
    """Keep overlapping boxes if there is 100% horizontal or vertical overlap.

    This is used to handle edge cases where un-nesting with a soft margin leaves
    ambiguously overlapping boxes. In this case, we choose to keep both boxes,
    accepting the risk of duplicate text instead of the higher risk of low recall
    text blocks that don't make sense.

    Args:
        box_1: The first box to compare.
        box_2: The second box to compare.

    Returns:
        True if the boxes fully overlap horizontally or vertically, False otherwise.
    """

    line_1_horizontal_interval = (box_1.coordinates[0], box_1.coordinates[2])
    line_2_horizontal_interval = (box_2.coordinates[0], box_2.coordinates[2])
    line_1_vertical_interval = (box_1.coordinates[1], box_1.coordinates[3])
    line_2_vertical_interval = (box_2.coordinates[1], box_2.coordinates[3])
    if check_line_contained(
        line_1_horizontal_interval, line_2_horizontal_interval
    ) or check_line_contained(line_1_vertical_interval, line_2_vertical_interval):
        return True
    return False


def reduce_all_overlapping_boxes(
    blocks: Layout,
    min_overlapping_pixels_vertical: int = 5,
    min_overlapping_pixels_horizontal: int = 5,
    reduction_direction: str = "vertical",
) -> Layout:
    """Eliminate all overlapping boxes by reducing their size by the minimal amount necessary.

    In general, for every pair of rectangular boxes with coordinates of
    the form (x_top_left, y_top_left, x_bottom_right, y_bottom_right),
    we want to reshape them to eliminate the intersecting regions in the
    alignment direction. For example, if we want to eliminate overlaps of
    the following two rectangles with a prior that vertical overlaps should
    be removed, the transformation should be

    [(0,0,3,3),(1,1,2,4)] -> [(0,0,3,1), (1,3,2,4)]

    Args:
        blocks: The blocks to reduce.
        reduction_direction: The direction to reduce the boxes in.
        min_overlapping_pixels_vertical: The minimal pixel overlap needed to reduce boxes in vertical direction.
        min_overlapping_pixels_horizontal: The minimal pixel overlap needed to reduce boxes in horizontal direction.

    Returns:
        The new layout with blocks having no overlapping coordinates.
    """
    assert reduction_direction in (
        "vertical",
        "horizontal",
    ), "Invalid direction. Must be 'vertical' or 'horizontal'."
    edited_blocks = []
    edited_coords = []
    for i, box_1 in enumerate(blocks):
        for j, box_2 in enumerate(blocks):
            if i == j:
                continue
            if check_horizontal_or_vertical_overlap(box_1, box_2):
                edited_blocks.append(box_1)
                edited_coords.append(box_2)
            else:
                intersection_area = box_1.intersect(box_2).area
                if intersection_area > 0:
                    if reduction_direction == "vertical":
                        if box_1.coordinates[3] < box_2.coordinates[3]:
                            rect_1, rect_2 = reduce_overlapping_boxes(
                                box_1,
                                box_2,
                                direction=reduction_direction,
                                min_overlapping_pixels_vertical=min_overlapping_pixels_vertical,
                            )
                        elif box_1.coordinates[3] == box_2.coordinates[3]:
                            continue
                        else:
                            rect_1, rect_2 = reduce_overlapping_boxes(
                                box_2,
                                box_1,
                                direction=reduction_direction,
                                min_overlapping_pixels_vertical=min_overlapping_pixels_vertical,
                            )
                    elif reduction_direction == "horizontal":
                        if box_1.coordinates[0] < box_2.coordinates[0]:
                            rect_1, rect_2 = reduce_overlapping_boxes(
                                box_1,
                                box_2,
                                direction=reduction_direction,
                                min_overlapping_pixels_horizontal=min_overlapping_pixels_horizontal,
                            )
                        elif box_1.coordinates[0] == box_2.coordinates[0]:
                            continue
                        else:
                            rect_1, rect_2 = reduce_overlapping_boxes(
                                box_2,
                                box_1,
                                direction=reduction_direction,
                                min_overlapping_pixels_horizontal=min_overlapping_pixels_horizontal,
                            )
                    box_1.box = rect_1
                    box_2.box = rect_2
                    if box_1.coordinates not in edited_coords:
                        edited_blocks.append(box_1)
                        edited_coords.append(box_1.coordinates)
                    if box_2.coordinates not in edited_coords:
                        edited_blocks.append(box_2)
                        edited_coords.append(box_2.coordinates)
                else:
                    if box_1.coordinates not in edited_coords:
                        edited_blocks.append(box_1)
                        edited_coords.append(box_1.coordinates)
    return Layout(edited_blocks)


def get_all_nested_blocks(
    layout: Layout, soft_margin: dict
) -> List[Tuple[int, TextBlock, int, TextBlock]]:
    """Get all nested blocks in the layout (blocks within other blocks)."""
    nested_block_indices = []
    for ix_1, box_1 in enumerate(layout):
        for ix_2, box_2 in enumerate(layout):
            if box_1 != box_2 and box_1.is_in(box_2, soft_margin):
                nested_block_indices.append((ix_1, box_1, ix_2, box_2))
    return nested_block_indices


def remove_contained_boxes(
    layout_: Layout, soft_margin: dict, max_recursion_count: int, counter: int
) -> Layout:
    """
    Remove all contained boxes from the layout.

    Identify the indices of the contained blocks and remove the ones with lower confidence scores.
    Continue this process recursively until there are no more contained blocks.
    """
    _LOGGER.debug(
        "Removing contained boxes.",
        extra={
            "props": {
                "layout": layout_,
                "soft_margin": soft_margin,
                "counter": counter,
            }
        },
    )

    if counter == max_recursion_count:
        _LOGGER.debug(
            "Max recursion depth reached.",
            extra={
                "props": {
                    "counter": counter,
                }
            },
        )
        return layout_

    nested_blocks = get_all_nested_blocks(layout_, soft_margin)
    if nested_blocks == []:
        _LOGGER.debug(
            "No nested blocks found.",
            extra={
                "props": {
                    "nested_indices": nested_blocks,
                }
            },
        )
        return layout_

    indices_to_remove = []
    for ix1, box1, ix2, box2 in nested_blocks:
        if box1.score > box2.score:
            indices_to_remove.append(box2)
        else:
            indices_to_remove.append(box1)

    _LOGGER.debug(
        "Filtering out the identified nested blocks.",
        extra={
            "props": {
                "indices_to_remove": indices_to_remove,
            }
        },
    )
    layout_ = Layout(
        [box for index, box in enumerate(layout_) if index not in indices_to_remove]
    )

    remove_contained_boxes(layout_, soft_margin, max_recursion_count, counter + 1)
    return layout_


def remove_nested_boxes(layout: Layout, un_nest_soft_margin: int = 15) -> Layout:
    """
    If a box is entirely or mostly contained within another box, remove the inner box.

    Args:
        layout: The layout to un nest.
        un_nest_soft_margin: The number of pixels to inflate each box by in each direction
        when checking for containment (i.e. a soft margin).
    Returns:
        The layout of un nested boxes.
    """
    if len(layout) == 0:
        return layout

    soft_margin = {
        "top": un_nest_soft_margin,
        "bottom": un_nest_soft_margin,
        "left": un_nest_soft_margin,
        "right": un_nest_soft_margin,
    }

    return remove_contained_boxes(layout, soft_margin, len(layout), 0)


# TODO: This is not part of the module and is for a CLI, but placing here for visibility before I edit the CLI.
def run_disambiguation_pipeline(
    image: PpmImageFile,
    model: Detectron2LayoutModel,
    restrictive_model_threshold: float,
    unnest_soft_margin: int,
    min_overlapping_pixels_horizontal: int,
    min_overlapping_pixels_vertical: int,
    combination_threshold: float = 0.8,
) -> Layout:
    """
    Initial output from layoutparser is ambiguous (overlapping boxes, nested boxes, etc). Disambiguate.

    Args: image: An image of a PDF page to perform object/block detection on. model: Object detection model.
    restrictive_model_threshold: Model confidence to separate blocks into two groups, a "restrictive" group and a
    "non-restrictive" group.
    unnest_soft_margin: The number of pixels to inflate each box by in each direction when
    checking for containment (i.e. a soft margin).
    min_overlapping_pixels_vertical: The min number of overlapping pixels before reducing boxes in vertical direction.
    min_overlapping_pixels_horizontal: The min number of overlapping pixels before reducing boxes in horizontal direction.

    Returns:
        A layout object containing only blocks from layoutparser with best effort disambiguation.
    """
    layout_unfiltered = Layout([b for b in model.detect(image)])
    layout_unnested = remove_nested_boxes(
        layout_unfiltered, un_nest_soft_margin=unnest_soft_margin
    )
    layout_restrictive, layout_permissive = split_layout(
        layout_unnested, restrictive_model_threshold=restrictive_model_threshold
    )
    layout_permissive = calculate_unexplained_fractions(
        layout_restrictive, layout_permissive
    )
    layout_combined = combine_layouts(
        layout_restrictive,
        layout_permissive,
        combination_threshold=combination_threshold,
    )
    layout_vertically_reduced = reduce_all_overlapping_boxes(
        layout_combined,
        min_overlapping_pixels_vertical=min_overlapping_pixels_vertical,
        reduction_direction="vertical",
    )
    layout_all_reduced = reduce_all_overlapping_boxes(
        layout_vertically_reduced,
        min_overlapping_pixels_horizontal=min_overlapping_pixels_horizontal,
        reduction_direction="horizontal",
    )
    return layout_all_reduced
