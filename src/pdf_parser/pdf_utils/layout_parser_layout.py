import logging

from layoutparser import Layout

_LOGGER = logging.getLogger(__name__)


def get_layout_parser_layout(model, image) -> Layout:
    """Get the layout parser layout for a pdf file."""
    _LOGGER.debug("Getting layout parser layout for image.")
    layout = Layout([b for b in model.detect(image)])
    _LOGGER.debug(
        "Layout parser layout detected.",
        extra={
            "props": {
                "layout_length": len(layout),
            }
        },
    )

    return layout
