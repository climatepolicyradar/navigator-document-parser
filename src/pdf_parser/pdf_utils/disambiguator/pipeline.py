from layoutparser import Layout
import logging

from src.pdf_parser.pdf_utils.disambiguator.nested import remove_nested_boxes
from src.pdf_parser.pdf_utils.disambiguator.overlapping import reduce_overlapping_boxes
from src.pdf_parser.pdf_utils.disambiguator.unexplained import (
    calculate_unexplained_fractions,
)
from src.pdf_parser.pdf_utils.disambiguator.utils import split_layout, combine_layouts

_LOGGER = logging.getLogger(__name__)


def run_disambiguation(
    layout: Layout,
    restrictive_model_threshold,
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
    _LOGGER.debug(
        "Disambiguation pipeline running over layout.",
        extra={
            "props": {
                "layout_length": len(layout),
            }
        },
    )

    layout = remove_nested_boxes(layout, unnest_soft_margin=unnest_soft_margin)
    _LOGGER.debug(
        "Layout unnested.",
        extra={
            "props": {
                "layout_length": len(layout),
            }
        },
    )

    layout_restrictive, layout_permissive = split_layout(
        layout, restrictive_model_threshold=restrictive_model_threshold
    )
    _LOGGER.debug(
        "Layout split into permissive and restrictive blocks.",
        extra={
            "props": {
                "layout_restrictive_length": len(layout_restrictive),
                "layout_permissive_length": len(layout_permissive),
            }
        },
    )

    layout_permissive, unexplained_fractions = calculate_unexplained_fractions(
        layout_restrictive, layout_permissive
    )
    _LOGGER.debug(
        "Layout permissive unexplained fractions calculated.",
        extra={
            "props": {
                "layout_permissive_length": len(layout_permissive),
            }
        },
    )

    layout_combined = combine_layouts(
        layout_restrictive,
        layout_permissive,
        unexplained_fractions,
        combination_threshold=combination_threshold,
    )
    _LOGGER.debug(
        "Layout combined.",
        extra={
            "props": {
                "layout_combined_length": len(layout_combined),
            }
        },
    )

    layout_combined = reduce_overlapping_boxes(
        layout_combined,
        unnest_soft_margin=unnest_soft_margin,
        min_pixel_overlap_vertical=min_overlapping_pixels_vertical,
        min_pixel_overlap_horizontal=min_overlapping_pixels_horizontal,
    )
    _LOGGER.debug(
        "Overlapping reduced for the combined layout.",
        extra={
            "props": {
                "layout_combined_length": len(layout_combined),
            }
        },
    )

    return layout_combined
