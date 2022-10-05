import os
from typing import List


def _convert_to_bool(x: str) -> bool:
    if x.lower() == "true":
        return True
    elif x.lower() == "false":
        return False

    raise ValueError(f"Cannot convert {x} to bool. Input must be 'True' or 'False'.")


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
TEST_RUN = _convert_to_bool(str(os.getenv("TEST_RUN", True)))
RUN_PDF_PARSER = _convert_to_bool(str(os.getenv("RUN_PDF_PARSER", True)))
RUN_HTML_PARSER = _convert_to_bool(str(os.getenv("RUN_HTML_PARSER", True)))
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
QUEUE_READ_BATCH_SIZE = int(os.getenv("QUEUE_READ_BATCH_SIZE", 10))
QUEUE_CREATE_TIMEOUT = int(os.getenv("QUEUE_CREATE_TIMEOUT", 61))
QUEUE_SEND_MESSAGE_DELAY = int(os.getenv("QUEUE_SEND_MESSAGE_DELAY", 10))
# TODO: http request headers?
