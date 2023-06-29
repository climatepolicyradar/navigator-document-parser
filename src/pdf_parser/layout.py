from io import BytesIO

import cv2
import layoutparser as lp
import numpy as np
from PIL import Image
from layoutparser.elements.layout import Layout

from src.config import LAYOUTPARSER_BOX_DETECTION_THRESHOLD


def create_image_from_jpeg_bytes(jpeg_bytes: bytes) -> np.ndarray:
    """Converts a JPEG byte array into a numpy array and then decodes using cv2."""
    np_arr = np.frombuffer(jpeg_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def convert_bytes_to_jpeg(image_bytes) -> bytes:
    """Converts a byte array into a JPEG byte array."""
    image = Image.open(BytesIO(image_bytes))

    output = BytesIO()
    image.save(output, format="JPEG")
    return output.getvalue()


# TODO Test different thresholds, could even try and assign and vary the threshold
class LayoutParserWrapper:
    def __init__(self):
        self.model = lp.Detectron2LayoutModel(
            "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
            extra_config=[
                "MODEL.ROI_HEADS.SCORE_THRESH_TEST",
                LAYOUTPARSER_BOX_DETECTION_THRESHOLD,
            ],
            label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
        )

    def get_layout(self, image_content: bytes) -> Layout:
        return self.model.detect(
            create_image_from_jpeg_bytes(convert_bytes_to_jpeg(image_content))
        )


# TODO need to return the type here as well
def get_layout_parser_blocks(image_content: bytes, lp_obj: LayoutParserWrapper) -> list:
    """Returns a list of coordinates for each block in the layout."""
    return lp_obj.get_layout(image_content)._blocks
