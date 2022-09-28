import concurrent.futures
import logging
import multiprocessing
import time
import warnings
from functools import partial
from pathlib import Path
import hashlib
from typing import List, Union
import tempfile
import os

import requests
import fitz
import layoutparser as lp
import numpy as np
from fitz.fitz import EmptyFileError
from tqdm import tqdm
from cloudpathlib import S3Path
import psutil

from src.pdf_parser.pdf_utils.parsing_utils import (
    OCRProcessor,
    LayoutDisambiguator,
    PostProcessor,
)
from src import config

from src.base import ParserOutput, PDFPageMetadata, PDFData, ParserInput


class TqdmLoggingHandler(logging.Handler):
    """Handler for logging to tqdm"""

    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        """Emit a log message"""
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception as e:
            print(f"Error emitting tqdm logging handler: {e}")
            self.handleError(record)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(TqdmLoggingHandler())


def download_pdf(
    parser_input: ParserInput, output_dir: Union[Path, str]
) -> Path or None:
    """
    Get a PDF from a URL in a ParserInput object.

    :param: parser input
    :param: directory to save the PDF to
    :return: path to PDF file in output_dir
    """
    try:
        response = requests.get(parser_input.document_url)
    except Exception as e:
        logging.error(
            f"Could not fetch {parser_input.document_url} for {parser_input.document_id}: {e}"
        )
        return None

    if response.status_code != 200:
        logging.exception(f"Could not get PDF from {parser_input.document_url}")
        return None

    if response.headers["Content-Type"] != "application/pdf":
        logging.exception(
            f"Content-Type is for {parser_input.document_id} ({parser_input.document_url}) is not PDF: {response.headers['Content-Type']}"
        )
        return None

    output_path = Path(output_dir) / f"{parser_input.document_id}.pdf"

    with open(output_path, "wb") as f:
        f.write(response.content)

    return output_path


def parse_file(
    input_task: ParserInput,
    model,
    model_threshold_restrictive: float,
    ocr_agent: str,
    output_dir: Union[Path, S3Path],
    device: str,
):
    """Parse an individual pdf file.

    Args:
        input_task (ParserInput): Class specifying location of the PDF and other data about the task.
        model (layoutparser.LayoutModel): Layout model to use for parsing.
        model_threshold_restrictive (float): Threshold to use for parsing.
        ocr_agent (src.pdf_utils.parsing_utils.OCRProcessor): OCR agent to use for parsing.
        output_dir (Path): Path to the output directory.
        device (str): Device to use for parsing.
    """

    # TODO: do we want to handle exceptions raised by get_pdf here?
    logging.info(
        f"Beginning parsing for: {input_task.document_id} at {input_task.document_url}."
    )
    logging.info(psutil.virtual_memory())

    with tempfile.TemporaryDirectory() as temp_output_dir:
        pdf_path = download_pdf(input_task, temp_output_dir)

        if pdf_path is None:
            logging.info(
                f"PDF path is None for: {input_task.document_url} at {temp_output_dir} as document couldn't be "
                f"downloaded, isn't content-type pdf or the response status code is not 200. "
            )
        else:
            page_layouts, pdf_images = lp.load_pdf(pdf_path, load_images=True)
            document_md5sum = hashlib.md5(pdf_path.read_bytes()).hexdigest()

            # FIXME: handle EmptyFileError here using _pdf_num_pages

            model = _get_detectron_model(model, device)
            if ocr_agent == "tesseract":
                ocr_agent = lp.TesseractAgent()
            elif ocr_agent == "gcv":
                ocr_agent = lp.GCVAgent()

            all_pages_metadata = []
            all_text_blocks = []

            for page_idx, image in tqdm(
                enumerate(pdf_images), total=len(pdf_images), desc=pdf_path.name
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
                postprocessor = PostProcessor(disambiguated_layout)
                ocr_blocks = postprocessor.postprocess()
                ocr_processor = OCRProcessor(
                    image=np.array(image),
                    page_number=page_idx,
                    layout=ocr_blocks,
                    ocr_agent=ocr_agent,
                )
                page_text_blocks = ocr_processor.process_layout()
                all_text_blocks += page_text_blocks

                page_dimensions = (
                    page_layouts[page_idx].page_data["width"],
                    page_layouts[page_idx].page_data["height"],
                )
                page_metadata = PDFPageMetadata(
                    dimensions=page_dimensions,
                    page_number=page_idx,
                )

                all_pages_metadata.append(page_metadata)

            document = ParserOutput(
                document_id=input_task.document_id,
                document_url=input_task.document_url,
                document_name=input_task.document_name,
                document_description=input_task.document_description,
                document_content_type=input_task.document_content_type,
                document_slug=input_task.document_slug,
                pdf_data=PDFData(
                    page_metadata=all_pages_metadata,
                    md5sum=document_md5sum,
                    text_blocks=all_text_blocks,
                ),
            ).set_document_languages_from_text_blocks(min_language_proportion=0.4)

            output_path = output_dir / f"{input_task.document_id}.json"

            output_path.write_text(document.json(indent=4, ensure_ascii=False))

            logging.info(f"Saved {output_path.name} to {output_dir}.")

            os.remove(pdf_path)

            logging.info(f"Removed downloaded document at - {pdf_path}.")


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


def run_pdf_parser(
    input_tasks: List[ParserInput],
    output_dir: Union[Path, S3Path],
    parallel: bool,
    device: str = "cpu",
) -> None:
    """
    Run cli to extract semi-structured JSON from document-AI + OCR.

    Args:
        input_tasks: List of tasks for the parser to process.
        output_dir: The directory to write the parsed PDFs to.
        parallel: Whether to run parsing over multiple processes.
        device: The device to use for the document AI model.
    """
    time_start = time.time()
    # ignore warnings that pollute the logs.
    warnings.filterwarnings("ignore")

    logging.info(
        f"Using {config.PDF_OCR_AGENT} OCR agent and {config.LAYOUTPARSER_MODEL} model."
    )
    if config.PDF_OCR_AGENT == "gcv":
        logging.warning(
            "THIS IS COSTING MONEY/CREDITS!!!! - BE CAREFUL WHEN TESTING. SWITCH TO TESSERACT (FREE) FOR TESTING."
        )

    logging.info("Iterating through files and parsing pdf content from pages.")
    # Sort pages smallest to largest. Having files of a similar size will help with parallelization.
    # FIXME: have had to disable this for now because we're using tasks, which don't have direct access to the PDF.
    # files = sorted(list(input_dir_path.glob("*.pdf")), key=_pdf_num_pages)

    file_parser = partial(
        parse_file,
        model=config.LAYOUTPARSER_MODEL,
        ocr_agent=config.PDF_OCR_AGENT,
        output_dir=output_dir,
        model_threshold_restrictive=config.LAYOUTPARSER_MODEL_THRESHOLD_RESTRICTIVE,
        device=device,
    )
    if parallel:
        cpu_count = multiprocessing.cpu_count() - 1
        with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
            executor.map(file_parser, tqdm(input_tasks))

    else:
        for task in tqdm(input_tasks):
            file_parser(task)

    logging.info("Finished parsing pdf content from pages.")
    time_end = time.time()
    logging.info(f"Time taken: {time_end - time_start} seconds.")
