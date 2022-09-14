import concurrent.futures
import logging
import multiprocessing
import time
import warnings
from functools import partial
from pathlib import Path
import hashlib
from typing import Callable

import click
import fitz
import layoutparser as lp
import numpy as np
from cloudpathlib import CloudPath
from fitz.fitz import EmptyFileError
from multiprocessing_logging import install_mp_handler
from tqdm import tqdm

from src.pdf_parser.pdf_utils.parsing_utils import (
    OCRProcessor,
    LayoutDisambiguator,
    DetectReadingOrder,
)

from src.base import PDFParserOutput, PDFPage


def parse_file(file, model, model_threshold_restrictive, ocr_agent, output_dir, device):
    """Parse an individual pdf file.

    Args:
        file (str): Path to the pdf file.
        model (layoutparser.LayoutModel): Layout model to use for parsing.
        model_threshold_restrictive (float): Threshold to use for parsing.
        ocr_agent (src.pdf_utils.parsing_utils.OCRProcessor): OCR agent to use for parsing.
        output_dir (str): Path to the output directory.
        device (str): Device to use for parsing.

    """

    model = _get_detectron_model(model, device)
    if ocr_agent == "tesseract":
        ocr_agent = lp.TesseractAgent(languages="eng")
    elif ocr_agent == "gcv":
        ocr_agent = lp.GCVAgent(languages="eng")

    page_layouts, pdf_images = lp.load_pdf(file, load_images=True)
    pages = []
    for page_idx, image in tqdm(
        enumerate(pdf_images), total=len(pdf_images), desc=file.name
    ):
        # Maybe we should always pass a layout object into the PageParser class.
        layout_disambiguator = LayoutDisambiguator(
            image, model, model_threshold_restrictive
        )
        initial_layout = layout_disambiguator.layout
        if len(initial_layout) == 0:
            logging.info(f"No layout found for page {page_idx}.")
            continue
        disambiguated_layout = layout_disambiguator.disambiguate_layout()
        reading_order_detector = DetectReadingOrder(disambiguated_layout)
        ocr_blocks = reading_order_detector.infer_reading_order()
        ocr_processor = OCRProcessor(
            image=np.array(image),
            page_number=page_idx,
            layout=ocr_blocks,
            ocr_agent=ocr_agent,
        )
        text_blocks = ocr_processor.process_layout()

        page_dimensions = (
            page_layouts[page_idx].page_data["width"],
            page_layouts[page_idx].page_data["height"],
        )
        page = PDFPage(
            text_blocks=text_blocks,
            dimensions=page_dimensions,
            page_number=page_idx,
        )

        pages.append(page)

    document = PDFParserOutput(
        # FIXME: Add ID based on input task
        id="",
        pages=pages,
        filename=file.stem,
        md5hash=hashlib.md5(file.read_bytes()).hexdigest(),
    ).set_languages(min_language_proportion=0.4)

    output_path = output_dir / f"{file.stem}.json"

    with open(output_path, "w") as f:
        f.write(document.json(indent=4, ensure_ascii=False))

    logging.info(f"Saved {output_path.name} to {output_dir}.")


def parse_all_files(files: list, func: Callable):
    """Parse all files in a list in parallel."""
    cpu_count = multiprocessing.cpu_count() - 1
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
        executor.map(func, files)


def _pdf_num_pages(file: str):
    """Get the number of pages in a pdf file."""
    try:
        return fitz.open(file).page_count
    except EmptyFileError:
        return 0


# TODO: We may want to make this an option, but for now just use Detectron by default as we are unlikely
#  to change this unless we start labelling by ourselves.
def _get_detectron_model(model: str, device: str) -> lp.Detectron2LayoutModel:
    return lp.Detectron2LayoutModel(
        config_path=f"lp://PubLayNet/{model}",  # In model catalog,
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
        device=device,
    )


@click.command()
@click.option(
    "-i",
    "--input-dir",
    type=str,
    required=True,
    help="The directory to read PDFs from.",
)
@click.option(
    "-o",
    "--output-dir",
    type=str,
    required=True,
)
@click.option(
    "--ocr-agent",
    type=click.Choice(["tesseract", "gcv"]),
    required=True,
    default="tesseract",
)
@click.option(
    "--device", type=click.Choice(["cuda", "cpu"]), required=True, default="cpu"
)
@click.option("--parallel", is_flag=True, default=False)
@click.option("-l", "--test-limit", type=int, default=1)
@click.option(
    "-m",
    "--model",
    type=str,
    required=True,
    help="The model to use for OCR.",
    default="mask_rcnn_X_101_32x8d_FPN_3x",  # powerful detectron-2 model.
)
@click.option(
    "-ts", "--model-threshold-restrictive", type=float, default=0.5
)  # Hyperparam, set up config
def run_cli(
    input_dir: str,
    output_dir: str,
    test_limit: int,
    ocr_agent: str,
    parallel: bool,
    model: str,
    model_threshold_restrictive: float = 0.4,
    device: str = "cpu",
) -> None:
    """
    Run cli to extract semi-structured JSON from document-AI + OCR.

    Args:
        input_dir: The directory containing the PDFs to parse.
        output_dir: The directory to write the parsed PDFs to.
        ocr_agent: The OCR agent to use.
        parallel: Whether to run parsing over multiple processes.
        test_limit: Place a limit on the number of PDFs to parse - useful for testing.
        model: The document AI model to use.
        model_threshold_restrictive: The threshold to use for the document AI model.
        device: The device to use for the document AI model.
    """
    time_start = time.time()
    # ignore warnings that pollute the logs.
    warnings.filterwarnings("ignore")
    # logging for multiprocessing.
    install_mp_handler()
    # Create logger that prints to stdout.
    logging.basicConfig(level=logging.DEBUG)

    logging.info(
        f"Test limit is {test_limit}. Using {ocr_agent} OCR agent and {model} model."
    )
    if ocr_agent == "gcv":
        logging.warning(
            "THIS IS COSTING MONEY/CREDITS!!!! - BE CAREFUL WHEN TESTING. SWITCH TO TESSERACT (FREE) FOR TESTING."
        )
    logging.info(f"Reading from {input_dir}.")
    if input_dir.startswith("s3://"):
        input_dir_path = CloudPath(input_dir)
    else:
        input_dir_path = Path(input_dir)

    if output_dir.startswith("s3://"):
        output_dir_path = CloudPath(output_dir)
    else:
        output_dir_path = Path(output_dir)

    logging.info("Iterating through files and parsing pdf content from pages.")
    # Sort pages smallest to largest. Having files of a similar size will help with parallelization.
    files = sorted(list(input_dir_path.glob("*.pdf")), key=_pdf_num_pages)
    files = [file for file in files if _pdf_num_pages(file) > 0]
    files = files[:test_limit]

    file_parser = partial(
        parse_file,
        model=model,
        ocr_agent=ocr_agent,
        output_dir=output_dir_path,
        model_threshold_restrictive=model_threshold_restrictive,
        device=device,
    )
    if parallel:
        parse_all_files(files, file_parser)
    else:
        for file in files:
            file_parser(file)
    logging.info("Finished parsing pdf content from pages.")
    time_end = time.time()
    logging.info(f"Time taken: {time_end - time_start} seconds.")


if __name__ == "__main__":
    run_cli()
