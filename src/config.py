import multiprocessing
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
    os.getenv("LAYOUTPARSER_MODEL_THRESHOLD_RESTRICTIVE", "0.4")
)

OCR_BLOCKS = [
    "Google Text Block",
    "Text",
    "List",
    "Title",
    "Ambiguous",
    "Inferred from gaps",
    "Table",
    "Figure",
]

# This is the number of pixels in the soft margin for a box to be considered nested within another box.
# In particular, we inflate the potential container box by this amount in each direction, and then
# check if the potential contained box is fully contained within the inflated container box.
LAYOUTPARSER_UNNEST_SOFT_MARGIN = int(
    os.getenv("LAYOUTPARSER_UNNEST_SOFT_MARGIN", "15")
)
# We want to avoid box overlaps to avoid OCR capturing cut-off text or capturing text twice.
# This is the minimum number of pixel overlaps before reducing size to avoid OCR conflicts. The
# idea is that a small overlap is probably just whitespace, but a large overlap is probably
# text that is being captured twice.
LAYOUTPARSER_MIN_OVERLAPPING_PIXELS_HORIZONTAL = int(
    os.getenv("LAYOUTPARSER_MIN_OVERLAPPING_PIXELS_HORIZONTAL", "5")
)
# similar to above, but for vertical overlaps
LAYOUTPARSER_MIN_OVERLAPPING_PIXELS_VERTICAL = int(
    os.getenv("LAYOUTPARSER_MIN_OVERLAPPING_PIXELS_VERTICAL", "5")
)
# Percentage of page to ignore at the top of the page when adding blocks to the
# page from google (e.g. to ignore headers).
LAYOUTPARSER_TOP_EXCLUDE_THRESHOLD = float(
    os.getenv("LAYOUTPARSER_TOP_EXCLUDE_THRESHOLD", "0.1")
)
# Percentage of page to ignore at the bottom of the page when adding blocks to the
# page from google (e.g. to ignore footers).
LAYOUTPARSER_BOTTOM_EXCLUDE_THRESHOLD = float(
    os.getenv("LAYOUTPARSER_BOTTOM_EXCLUDE_THRESHOLD", "0.1")
)
# Threshold for replacing blocks from google with blocks from the model. e.g.
# if a block from layoutparser is 95% covered by a block from google, as measured by intersection over
# union, then the block from layoutparser will be replaced by the block from google.
LAYOUTPARSER_REPLACE_THRESHOLD = float(
    os.getenv("LAYOUTPARSER_REPLACE_THRESHOLD", "0.9")
)
# The fraction of unexplained area from restrrctive layours above which to include boxes
# from the permissive layout.
LAYOUTPARSER_DISAMBIGUATION_COMBINATION_THRESHOLD = float(
    os.getenv("LAYOUTPARSER_DISAMBIGUATION_COMBINATION_THRESHOLD", "0.8")
)

PDF_OCR_AGENT = os.getenv("PDF_OCR_AGENT", "gcv")

TEST_RUN = os.getenv("TEST_RUN", "false").lower() == "true"
RUN_PDF_PARSER = os.getenv("RUN_PDF_PARSER", "true").lower() == "true"
RUN_HTML_PARSER = os.getenv("RUN_HTML_PARSER", "true").lower() == "true"
RUN_TRANSLATION = os.getenv("RUN_TRANSLATION", "true").lower() == "true"

# Default set by trial and error based on behaviour of the parsing model
PDF_N_PROCESSES = int(os.getenv("PDF_N_PROCESSES", multiprocessing.cpu_count() / 2))
FILES_TO_PARSE = os.getenv("files_to_parse")
LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "DEBUG").upper()
