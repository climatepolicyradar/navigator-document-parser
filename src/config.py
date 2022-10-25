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
PDF_OCR_AGENT = os.getenv("PDF_OCR_AGENT", "gcv")

TEST_RUN = os.getenv("TEST_RUN", "false").lower() == "true"
RUN_PDF_PARSER = os.getenv("RUN_PDF_PARSER", "true").lower() == "true"
RUN_HTML_PARSER = os.getenv("RUN_HTML_PARSER", "true").lower() == "true"
RUN_TRANSLATION = os.getenv("RUN_TRANSLATION", "true").lower() == "true"

# Default set by trial and error based on behaviour of the parsing model
PDF_N_PROCESSES = int(os.getenv("PDF_N_PROCESSES", multiprocessing.cpu_count() / 2))
FILES_TO_PARSE = os.getenv("files_to_parse")
LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "INFO")
