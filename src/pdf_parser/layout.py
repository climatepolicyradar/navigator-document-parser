import os
from io import BytesIO

import layoutparser as lp
import cv2
import numpy as np
from PIL import Image
from layoutparser.elements.layout import Layout


def create_image_from_jpeg_bytes(jpeg_bytes):
    # Convert JPEG bytes to numpy array
    np_arr = np.frombuffer(jpeg_bytes, np.uint8)
    # Decode the JPEG array using cv2
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image


def convert_bytes_to_jpeg(image_bytes):
    # Load image from byte array
    image = Image.open(BytesIO(image_bytes))

    # Convert image to JPEG format
    output = BytesIO()
    image.save(output, format='JPEG')
    jpeg_data = output.getvalue()

    return jpeg_data


# TODO Test different thresholds, could even try and assign and vary the threshold
class LayoutParserWrapper:
    def __init__(self):
        self.model = lp.Detectron2LayoutModel(
            'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
            label_map={
                0: "Text",
                1: "Title",
                2: "List",
                3: "Table",
                4: "Figure"
            }
        )

    def get_layout(self, image_content: bytes) -> Layout:
        return self.model.detect(create_image_from_jpeg_bytes(convert_bytes_to_jpeg(image_content)))



