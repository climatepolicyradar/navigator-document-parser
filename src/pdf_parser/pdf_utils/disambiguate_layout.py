from layoutparser import Layout, TextBlock, Rectangle, Detectron2LayoutModel
from PIL.PpmImagePlugin import PpmImageFile
from shapely.geometry import Polygon
from shapely.ops import unary_union
from typing import List, Tuple, Optional

from pydantic import BaseModel as PydanticBaseModel, Field


class BaseModel(PydanticBaseModel):
    """Base class for all models."""

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True


# TODO: I added this because I want to enforce that the unexplained fractions are in the same order as the boxes in
#  the layout without adding it as page metadata, as this would require formally writing checks
#   with assert. Is there a better way to do this or is this ok?
class LayoutWithFractions(BaseModel):
    """Layout with unexplained fractions added."""

    layout: Layout
    # unexplained fractions must be a list of floats between 0 and 1
    unexplained_fractions: List[float] = Field(..., ge=0, le=1)


def split_layout(
    layout: Layout, restrictive_model_threshold: float
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
    layout_restrictive: Layout, layout_permissive: LayoutWithFractions, threshold: float
) -> Layout:
    """Add unexplained text boxes to the strict layout to get a combined layout.

    Args:
        layout_restrictive: The layout with boxes above the restrictive threshold.
        layout_permissive: The layout with boxes below the restrictive threshold.
        threshold: The threshold above which to include boxes from the permissive layout.

    Returns:
        The layout with boxes from the unfiltered perspective added if their areas aren't already sufficiently accounted for..
    """
    boxes_to_add = []
    permissive_layout = layout_permissive.layout
    for ix, box in enumerate(permissive_layout):
        unexplained_fractions = layout_permissive.unexplained_fractions
        # If the box's area is not "explained away" by the strict layout, add it to the combined layout with an
        # ambiguous type tag. We can use heuristics to determine its type downstream.
        if unexplained_fractions[ix] > threshold:
            box.block_1.type = "Ambiguous"
            boxes_to_add.append(box)
    layout_combined = layout_restrictive + Layout(boxes_to_add)
    return layout_combined


def reduce_overlapping_boxes(
    box_1: TextBlock,
    box_2: TextBlock,
    direction: str = "vertical",
    max_overlapping_pixels_vertical: Optional[int] = 5,
    max_overlapping_pixels_horizontal: Optional[int] = 5,
) -> Tuple[TextBlock, TextBlock]:
    """Reduce the size of overlapping boxes to elimate overlaps.

    If two boxes overlap in a given direction (vertical or horizontal), reduce the size of both in that
     direction by the minimal amount necessary to elimate overlaps.

    Args:
        box_1: The first box to compare. This box should be the upper/left box.
        box_2: The second box to compare. This box should be the lower/right box.
        direction: The direction to reduce the boxes in.
        max_overlapping_pixels_vertical: The minimal pixel overlap needed to reduce boxes in vertical direction.
        max_overlapping_pixels_horizontal: The minimal pixel overlap needed to reduce boxes in horizontal direction.

    Returns:
        The boxes with overlaps eliminated.
    """

    if direction not in ("horizontal", "vertical"):
        raise ValueError("Direction must be 'horizontal' or 'vertical'.")

    if direction == "vertical":
        assert (
            box_1.block_1.coordinates[1] < box_2.block_1.coordinates[1]
        ), "box_1 should be the upper box."
        intersection_height = box_1.intersect(box_2).height
        if intersection_height > max_overlapping_pixels_vertical:
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
            box_1.block_1.coordinates[0] < box_2.block_1.coordinates[0]
        ), "box_1 should be the left box."
        intersection_width = box_1.intersect(box_2).width
        if intersection_width > max_overlapping_pixels_horizontal:
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


def reduce_all_overlapping_boxes(
    blocks: Layout,
    max_overlapping_pixels_vertical: int = 5,
    max_overlapping_pixels_horizontal: int = 5,
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

    Returns:
        The new layout with blocks having no overlapping coordinates.
    """
    assert reduction_direction in ("vertical", "horizontal"), (
        "Invalid direction. Must be 'vertical' or " "'horizontal'. "
    )
    for box_1 in blocks:
        for box_2 in blocks:
            if box_1 == box_2:
                continue
            else:
                intersection_area = box_1.intersect(box_2).area
                if intersection_area > 0:
                    if reduction_direction == "vertical":
                        # check which box is upper and which is lower
                        if box_1.coordinates[3] < box_2.coordinates[3]:
                            rect_1, rect_2 = reduce_overlapping_boxes(
                                box_1,
                                box_2,
                                direction=reduction_direction,
                                max_overlapping_pixels_vertical=max_overlapping_pixels_vertical,
                            )
                        else:
                            rect_1, rect_2 = reduce_overlapping_boxes(
                                box_2,
                                box_1,
                                direction=reduction_direction,
                                max_overlapping_pixels_vertical=max_overlapping_pixels_vertical,
                            )
                    elif reduction_direction == "horizontal":
                        # check which box is left and which is right
                        if box_1.coordinates[2] < box_2.coordinates[2]:
                            rect_1, rect_2 = reduce_overlapping_boxes(
                                box_1,
                                box_2,
                                direction=reduction_direction,
                                max_overlapping_pixels_horizontal=max_overlapping_pixels_horizontal,
                            )
                        else:
                            rect_1, rect_2 = reduce_overlapping_boxes(
                                box_2,
                                box_1,
                                direction=reduction_direction,
                                max_overlapping_pixels_horizontal=max_overlapping_pixels_horizontal,
                            )
                    box_1.block_1 = rect_1
                    box_2.block_1 = rect_2
    return blocks


def unnest_boxes(layout: Layout, unnest_soft_margin: int = 15) -> Layout:
    """
    Loop through boxes, unnest them until there are no nested boxes left..

    Args: layout: The layout to unnest.
    unnest_soft_margin: The number of pixels to inflate each box by in each direction
    when checking for containment (i.e. a soft margin).

    Returns:
        The unnested boxes.
    """
    if len(layout) == 0:
        return layout
    else:
        # Add a soft-margin for the is_in function to allow for some leeway in the containment check.
        soft_margin = {
            "top": unnest_soft_margin,
            "bottom": unnest_soft_margin,
            "left": unnest_soft_margin,
            "right": unnest_soft_margin,
        }
        # The loop checks each block for containment within other blocks.
        # Contained blocks are removed if they have lower confidence scores than their parents;
        # otherwise, the parent is removed. The process continues until there are no contained blocks.
        # There are potentially nestings within nestings, hence the complicated loop.
        # A recursion might be more elegant, leaving it as a TODO.
        stop_cond = True
        counter = 0  # num contained blocks in every run through of all pairs to calculate stop
        # condition.
        ixs_to_remove = []
        while stop_cond:
            for ix, box_1 in enumerate(layout):
                for ix2, box_2 in enumerate(layout):
                    if box_1 == box_2:
                        continue
                    else:
                        if box_1.is_in(box_2, soft_margin):
                            counter += 1
                            # Remove the box the model is less confident about.
                            if box_1.score > box_2.score:
                                remove_ix = ix2
                            else:
                                remove_ix = ix
                            ixs_to_remove.append(remove_ix)
                # stop condition: no contained blocks
                if counter == 0:
                    stop_cond = False
                counter = 0

        layout_unnested = Layout(
            [box for index, box in enumerate(layout) if index not in ixs_to_remove]
        )
        return layout_unnested


# TODO: This is not part of the module and is for a CLI, but placing here for visibility before I edit the CLI.
def disambiguation_pipeline(
    image: PpmImageFile,
    model: Detectron2LayoutModel,
    restrictive_model_threshold: float,
    unnest_soft_margin: int,
    max_overlapping_pixels_vertical: int,
    max_overlapping_pixels_horizontal: int,
) -> Layout:
    """
    Run the disambiguation pipeline on an image.

    @param image: An image of a PDF page to perform object/block detection on.
    @param model: Object detection model.
    @param restrictive_model_threshold: Model confidence to separate blocks into two groups, a "restrictive" group and a "non-restrictive" group.
    @param unnest_soft_margin: The number of pixels to inflate each box by in each direction when checking for containment for unnesting.
    @param max_overlapping_pixels_vertical: The maximum number of pixels to allow for vertical overlaps.
    @param max_overlapping_pixels_horizontal: The maximum number of pixels to allow for horizontal overlaps.
    @return: A layout object containing only blocks from layoutparser with best effort disambiguation..
    """
    layout_unfiltered = Layout([b for b in model.detect(image)])
    layout_unnested = unnest_boxes(
        layout_unfiltered, unnest_soft_margin=unnest_soft_margin
    )
    layout_restrictive, layout_permissive = split_layout(
        layout_unnested, restrictive_model_threshold=restrictive_model_threshold
    )
    layout_permissive = calculate_unexplained_fractions(
        layout_restrictive, layout_permissive
    )
    layout_vertically_reduced = reduce_all_overlapping_boxes(
        layout_permissive,
        max_overlapping_pixels_vertical=max_overlapping_pixels_vertical,
        reduction_direction="vertical",
    )
    layout_all_reduced = reduce_all_overlapping_boxes(
        layout_vertically_reduced,
        max_overlapping_pixels_horizontal=max_overlapping_pixels_horizontal,
        reduction_direction="horizontal",
    )
    return layout_all_reduced
