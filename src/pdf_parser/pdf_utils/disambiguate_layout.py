from layoutparser import Layout, TextBlock, Rectangle, Detectron2LayoutModel  # type: ignore
from PIL.PpmImagePlugin import PpmImageFile
from shapely.geometry import Polygon
from shapely.ops import unary_union
from typing import List, Tuple, Optional, Any


def split_layout_on_box_confidence(
    layout: Layout, restrictive_model_threshold: float = 0.5
) -> Tuple[Layout, Layout]:
    """
    Split layout into boxes above and below a given model confidence score.

    Args:
        layout: The layout to create polygons from.
        restrictive_model_threshold: The threshold above which to include boxes in the restrictive layout.

    Returns:
        A tuple of layouts, the first with boxes above the threshold and the second with boxes below the threshold.
    """
    restrictive_boxes = []
    permissive_boxes = []

    [
        restrictive_boxes.append(box)
        if box.score > restrictive_model_threshold
        else permissive_boxes.append(box)
        for box in layout
    ]

    return Layout(restrictive_boxes), Layout(permissive_boxes)


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


def get_not_covered_area(poly: Polygon, restrictive_polygons: List[Polygon]) -> float:
    """Get the area of a polygon not captured by all the restrictive layout polygons."""
    return poly.difference(unary_union(restrictive_polygons)).area


def get_permissive_area_fractions_not_in_restrictive(
    restrictive_layout: Layout, permissive_layout: Layout
) -> List[float]:
    """Calculate the fraction of each box in the permissive layout not captured by boxes in the restrictive layout.

    This is useful because we want to find boxes that are not already accounted for by the strict model but that may
    contain useful text.

    Args:
        restrictive_layout: The layout with boxes above the restrictive threshold.
        permissive_layout: The layout with boxes below the restrictive threshold.

    Returns:
        A LayoutWithFractions object containing a layout model.
    """
    restrictive_polygons = [
        lp_coords_to_shapely_polygon(box.coordinates) for box in restrictive_layout
    ]

    permissive_polygons = [
        lp_coords_to_shapely_polygon(box.coordinates) for box in permissive_layout
    ]

    return [
        get_not_covered_area(poly, restrictive_polygons) / poly.area
        for poly in permissive_polygons
    ]


def combine_layouts(
    layout_restrictive: Layout,
    layout_permissive: Layout,
    permissive_areas_not_covered_in_restrictive: List[float],
    combination_threshold: float,
) -> Layout:
    """Add unexplained text boxes to the strict layout to get a combined layout.

    Args: layout_restrictive: The layout with boxes above the restrictive threshold. layout_permissive: The layout
    with boxes below the restrictive threshold. permissive_areas_not_covered_in_restrictive: The fraction of each box
    in the permissive layout not captured by the restrictive areas. combination_threshold: The threshold above which
    to include boxes from the permissive layout.

    Returns: The layout with boxes from the unfiltered perspective added if their areas aren't already sufficiently
    accounted for.
    """
    boxes_to_add = [
        box
        for ix, box in enumerate(layout_permissive)
        if permissive_areas_not_covered_in_restrictive[ix] > combination_threshold
    ]
    for box in boxes_to_add:
        box.type = "Ambiguous"

    return layout_restrictive + Layout(boxes_to_add)


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
            box_1.coordinates[1] < box_2.coordinates[1]
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


def get_all_nested_block_indices(
    layout: Layout, soft_margin: dict
) -> List[Tuple[int, Any, int, Any]]:
    """Get all nested blocks in the layout (blocks within other blocks)."""
    nested_block_indices = []
    for ix_1, box_1 in enumerate(layout):
        for ix_2, box_2 in enumerate(layout):
            if box_1 != box_2 and box_1.is_in(box_2, soft_margin):
                nested_block_indices.append((ix_1, box_1, ix_2, box_2))
    return nested_block_indices


def remove_contained_boxes(layout_to_un_nest: Layout, soft_margin: dict) -> Layout:
    """
    Remove all contained boxes from the layout.

    Identify the indices of the contained blocks and remove the ones with lower confidence scores.
    Continue this process recursively until there are no more contained blocks.
    """
    nested_indices = get_all_nested_block_indices(layout_to_un_nest, soft_margin)
    if nested_indices is not []:
        for ix1, box1, ix2, box2 in nested_indices:
            if box1.score > box2.score:
                layout_to_un_nest.pop(ix2)
            else:
                layout_to_un_nest.pop(ix1)
        remove_contained_boxes(layout_to_un_nest, soft_margin)
    return layout_to_un_nest


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

    return remove_contained_boxes(layout, soft_margin)


# TODO: This is not part of the module and is for a CLI, but placing here for visibility before I edit the CLI.
def run_disambiguation_pipeline(
    image: PpmImageFile,
    model: Detectron2LayoutModel,
    restrictive_model_threshold: float,
    un_nest_soft_margin: int,
    min_overlapping_pixels_horizontal: int,
    min_overlapping_pixels_vertical: int,
    combination_threshold: float = 0.8,
) -> Layout:
    """
    Initial output from layoutparser is ambiguous (overlapping boxes, nested boxes, etc). Disambiguate.

    Args: image: An image of a PDF page to perform object/block detection on. model: Object detection model.
    restrictive_model_threshold: Model confidence to separate blocks into two groups, a "restrictive" group and a
    "non-restrictive" group.
    un_nest_soft_margin: The number of pixels to inflate each box by in each direction when
    checking for containment (i.e. a soft margin).
    min_overlapping_pixels_vertical: The min number of overlapping pixels before reducing boxes in vertical direction.
    min_overlapping_pixels_horizontal: The min number of overlapping pixels before reducing boxes in horizontal direction.

    Returns:
        A layout object containing only blocks from layoutparser with the best effort disambiguation.
    """
    layout = Layout([b for b in model.detect(image)])

    layout = remove_nested_boxes(layout=layout, un_nest_soft_margin=un_nest_soft_margin)

    layout_restrictive, layout_permissive = split_layout_on_box_confidence(
        layout=layout, restrictive_model_threshold=restrictive_model_threshold
    )

    permissive_areas_not_in_restrictive = (
        get_permissive_area_fractions_not_in_restrictive(
            layout_restrictive, layout_permissive
        )
    )

    layout_combined = combine_layouts(
        layout_permissive=layout_permissive,
        layout_restrictive=layout_restrictive,
        permissive_areas_not_covered_in_restrictive=permissive_areas_not_in_restrictive,
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
