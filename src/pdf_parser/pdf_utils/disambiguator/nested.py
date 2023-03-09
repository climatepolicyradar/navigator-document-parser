import itertools

from layoutparser import Layout
import logging

from src.pdf_parser.pdf_utils.disambiguator.utils import is_in

_LOGGER = logging.getLogger(__name__)


def remove_nested_boxes(layout: Layout, unnest_soft_margin: int = 15) -> Layout:
    """
    Loop through boxes, unnest them until there are no nested boxes left..

    Args: layout: The layout to unnest.
    unnest_soft_margin: The number of pixels to inflate each box by in each direction
    when checking for containment (i.e. a soft margin).

    Returns:
        The unnested boxes.
    """
    layout_length = len(layout)
    _LOGGER.info(
        "Unnesting boxes...",
        extra={
            "props": {
                "unnest_soft_margin": unnest_soft_margin,
                "layout_length": layout_length,
            }
        },
    )
    if len(layout) == 0:
        _LOGGER.info("No boxes to unnest.")
        return layout

    soft_margin = {
        "top": unnest_soft_margin,
        "bottom": unnest_soft_margin,
        "left": unnest_soft_margin,
        "right": unnest_soft_margin,
    }

    boxes_to_remove = []
    for box1, box2 in itertools.combinations(layout, 2):
        if box1.intersect(box2).area > 0:
            if is_in(box1, box2, soft_margin):
                if box1.score > box2.score:
                    boxes_to_remove.append(box2)
                else:
                    boxes_to_remove.append(box1)
            else:
                if is_in(box2, box1, soft_margin):
                    if box1.score > box2.score:
                        boxes_to_remove.append(box2)
                    else:
                        boxes_to_remove.append(box1)

    unnested_layout = Layout([box for box in layout if box not in boxes_to_remove])
    _LOGGER.info(
        "Unnested boxes.",
        extra={
            "props": {
                "unnest_soft_margin": unnest_soft_margin,
                "layout_length": layout_length,
                "unnested_layout_length": len(unnested_layout),
            }
        },
    )
    return unnested_layout
