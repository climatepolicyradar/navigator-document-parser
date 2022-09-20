import os
from typing import List

HTML_MIN_NO_LINES_FOR_VALID_TEXT = int(
    os.getenv("HTML_MIN_NO_LINES_FOR_VALID_TEXT", "6")
)
HTML_HTTP_REQUEST_TIMEOUT = int(os.getenv("HTML_HTTP_REQUEST_TIMEOUT", "30"))  # seconds
HTML_MAX_PARAGRAPH_LENGTH_WORDS = int(
    os.getenv("HTML_MAX_PARAGRAPH_LENGTH_WORDS", "500")
)
TARGET_LANGUAGES: List[str] = (
    os.getenv("TARGET_LANGUAGES", "en").lower().split(",")
)  # comma-separated 2-letter ISO codes
LAYOUTPARSER_MODEL = os.getenv("LAYOUTPARSER_MODEL", "mask_rcnn_X_101_32x8d_FPN_3x")
LAYOUTPARSER_MODEL_THRESHOLD_RESTRICTIVE = float(
    os.getenv("LAYOUTPARSER_MODEL_THRESHOLD_RESTRICTIVE", "0.5")
)
PDF_OCR_AGENT = os.getenv("PDF_OCR_AGENT", "gcv")


# TODO: http request headers?
