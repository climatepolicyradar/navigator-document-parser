import os
from io import BytesIO

import layoutparser as lp
import cv2
import numpy as np
from PIL import Image
from layoutparser.elements.layout import Layout


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
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
            label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
        )

    def get_layout(self, image_content: bytes) -> Layout:
        return self.model.detect(
            create_image_from_jpeg_bytes(convert_bytes_to_jpeg(image_content))
        )
