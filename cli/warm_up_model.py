"""This module presents a method for downloading the model that is to be used in the parser so that it can be
downloaded once and form a layer of the docker image rather than being downloaded at run time every time the
parser is instantiated. """

import logging
from typing import Union
import os

from layoutparser.models import Detectron2LayoutModel
from layoutparser.ocr import TesseractAgent, GCVAgent

_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.DEBUG)

PDF_OCR_AGENT = os.getenv("PDF_OCR_AGENT", "gcv")
LAYOUTPARSER_MODEL = os.getenv("LAYOUTPARSER_MODEL", "mask_rcnn_X_101_32x8d_FPN_3x")


def _get_detectron_model(model: str, device: str) -> Detectron2LayoutModel:
    return Detectron2LayoutModel(
        config_path=f"lp://PubLayNet/{model}",  # In model catalog,
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
        device=device,
    )


def get_model(
    model_name: str,
    ocr_agent_name: str,
    device: str,
) -> tuple[Detectron2LayoutModel, Union[TesseractAgent, GCVAgent]]:
    """Get the model for the parser."""
    _LOGGER.info(
        "Model Configuration",
        extra={
            "props": {
                "model": model_name,
                "ocr_agent": ocr_agent_name,
                "device": device,
            }
        },
    )
    if ocr_agent_name == "gcv":
        _LOGGER.warning(
            "THIS IS COSTING MONEY/CREDITS!!!! - BE CAREFUL WHEN TESTING. SWITCH TO TESSERACT (FREE) FOR TESTING."
        )

    model = _get_detectron_model(model_name, device)
    if ocr_agent_name == "tesseract":
        ocr_agent = TesseractAgent()
    elif ocr_agent_name == "gcv":
        ocr_agent = GCVAgent()
    else:
        raise RuntimeError(f"Uknown OCR agent type: '{ocr_agent_name}'")

    return model, ocr_agent


if __name__ == "__main__":
    get_model(
        model_name=LAYOUTPARSER_MODEL,
        ocr_agent_name=PDF_OCR_AGENT,
        device="cpu",
    )
