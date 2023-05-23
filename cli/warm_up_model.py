"""Module presents a method for downloading the model that is to be used in the parser.

This is so that it can be downloaded once and form a layer of the docker image rather than being downloaded at
run time every time the parser is instantiated.
"""

import os

# Set cdn domain as the environment variable is required by the run_parser.py script
os.environ["CDN_DOMAIN"] = "cdn.dummy.com"

from cli.parse_pdfs import get_model  # noqa: E402
from src.config import LAYOUTPARSER_MODEL, PDF_OCR_AGENT  # noqa: E402


if __name__ == "__main__":
    get_model(
        model_name=LAYOUTPARSER_MODEL,
        ocr_agent_name=PDF_OCR_AGENT,
        device="cpu",
    )
