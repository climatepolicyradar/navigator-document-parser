import os
from typing import List
import multiprocessing

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
PDF_OCR_AGENT = os.getenv("PDF_OCR_AGENT", "tesseract")

TEST_RUN = os.getenv("TEST_RUN", False)
RUN_PDF_PARSER = os.getenv("RUN_PDF_PARSER", True)
RUN_HTML_PARSER = os.getenv("RUN_HTML_PARSER", True)

# Default set by trial and error based on behaviour of the parsing model
PDF_N_PROCESSES = int(os.getenv("PDF_N_PROCESSES", multiprocessing.cpu_count() / 2))

# TODO: http request headers?
