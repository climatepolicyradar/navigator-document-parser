import concurrent.futures
import hashlib
import logging
import multiprocessing
import os
import tempfile
import time
import warnings
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import List, Optional, Union

import fitz
import layoutparser as lp
import numpy as np
import requests
from cloudpathlib import CloudPath, S3Path
from fitz.fitz import EmptyFileError
from tqdm import tqdm

from src import config
from src.base import (
    ParserInput,
    ParserOutput,
    PDFData,
    PDFPageMetadata,
    StandardErrorLog,
)
from src.pdf_parser.pdf_utils.parsing_utils import (
    LayoutDisambiguator,
    OCRProcessor,
    PostProcessor,
)

CDN_DOMAIN = os.environ["CDN_DOMAIN"]


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def copy_input_to_output_pdf(
    task: ParserInput, output_path: Union[Path, CloudPath]
) -> None:
    """Necessary to copy the input file to the output to ensure that we don't drop documents.

    The file is copied at the time of processing rather than syncing the entire input directory so that if that
    parser fails and retries it will not think that all files have already been parsed. :param task: input task
    specifying the document to copy :param output_path: path to save the copied file
    """
    try:
        blank_output = ParserOutput(
            document_id=task.document_id,
            document_metadata=task.document_metadata,
            document_name=task.document_name,
            document_description=task.document_description,
            document_source_url=task.document_source_url,
            document_cdn_object=task.document_cdn_object,
            document_md5_sum=task.document_md5_sum,
            document_slug=task.document_slug,
            document_content_type=task.document_content_type,
            languages=None,
            translated=False,
            html_data=None,
            pdf_data=PDFData(page_metadata=[], md5sum="", text_blocks=[]),
        )

        output_path.write_text(blank_output.json(indent=4, ensure_ascii=False))
        logging.info(f"Blank output for {task.document_id} saved to {output_path}.")

    except Exception as e:
        logger.error(
            StandardErrorLog.parse_obj(
                {
                    "timestamp": datetime.now(),
                    "pipeline_stage": "Parser: Copy pdf input to output.",
                    "status_code": "None",
                    "error_type": "ParsingError",
                    "message": f"{e}",
                    "document_in_process": output_path,
                }
            )
        )


def download_pdf(
    parser_input: ParserInput,
    output_dir: Union[Path, str],
) -> Optional[Path]:
    """
    Get a PDF from a URL in a ParserInput object.

    :param: parser input
    :param: directory to save the PDF to
    :return: path to PDF file in output_dir
    """
    try:
        document_url = f"https://{CDN_DOMAIN}/{parser_input.document_cdn_object}"
        logger.info(f"Downloading {document_url} to {output_dir}")
        response = requests.get(document_url)
    except Exception as e:
        logger.error(
            StandardErrorLog.parse_obj(
                {
                    "timestamp": datetime.now(),
                    "pipeline_stage": "Parser: Download pdf",
                    "status_code": "None",
                    "error_type": "RequestError",
                    "message": f"{e}",
                    "document_in_process": str(parser_input.document_id),
                }
            )
        )
        return None

    if response.status_code != 200:
        logger.error(
            StandardErrorLog.parse_obj(
                {
                    "timestamp": datetime.now(),
                    "pipeline_stage": "Parser: Download of pdf.",
                    "status_code": f"{response.status_code}",
                    "error_type": "RequestError",
                    "message": "Invalid response code from request.",
                    "document_in_process": str(parser_input.document_id),
                }
            )
        )

        return None

    elif response.headers["Content-Type"] != "application/pdf":
        logger.error(
            StandardErrorLog.parse_obj(
                {
                    "timestamp": datetime.now(),
                    "pipeline_stage": "Parser: Validate Content-Type of downloaded file.",
                    "status_code": f"{response.status_code}",
                    "error_type": "ContentTypeError",
                    "message": "Content-Type is not application/pdf.",
                    "document_in_process": str(parser_input.document_id),
                }
            )
        )

        return None

    else:
        logger.info(f"Saving {parser_input.document_source_url} to {output_dir}")
        output_path = Path(output_dir) / f"{parser_input.document_id}.pdf"

        with open(output_path, "wb") as f:
            f.write(response.content)

        return output_path


def select_page_at_random(num_pages: int) -> bool:
    """Determine whether to include a page using a random number generator. Used for debugging.

    Args:
        num_pages: The number of pages in the PDF.

    Returns:
        The page number to include.
    """
    rng = np.random.random()
    if num_pages in range(1, 10):
        # Only include pages at random for debugging to dramatically speed up processing (some PDFs have 100s
        # of pages)
        if rng < 0.5:
            return True
    elif num_pages in range(10, 100):
        if rng < 0.1:
            return True
    else:
        if rng <= 0.05:
            return True
    return False


def parse_file(
    input_task: ParserInput,
    model,
    model_threshold_restrictive: float,
    ocr_agent: str,
    debug: bool,
    output_dir: Union[Path, S3Path],
    device: str,
):
    """Parse an individual pdf file.

    Args:
        input_task (ParserInput): Class specifying location of the PDF and other data about the task.
        model (layoutparser.LayoutModel): Layout model to use for parsing.
        model_threshold_restrictive (float): Threshold to use for parsing.
        debug (bool): Whether to save debug images.
        ocr_agent (src.pdf_utils.parsing_utils.OCRProcessor): OCR agent to use for parsing.
        output_dir (Path): Path to the output directory.
        device (str): Device to use for parsing.
    """

    logging.info(f"Processing {input_task.document_id}")
    copy_input_to_output_pdf(input_task, output_dir / f"{input_task.document_id}.json")

    with tempfile.TemporaryDirectory() as temp_output_dir:
        logging.info(f"Downloading pdf: {input_task.document_id}")
        pdf_path = download_pdf(input_task, temp_output_dir)
        logging.info(f"PDF path for: {input_task.document_id} - {pdf_path}")
        if pdf_path is None:
            logging.info(
                f"PDF path is None for: {input_task.document_id} at {temp_output_dir} as document couldn't be "
                f"downloaded, isn't content-type pdf or the response status code is not 200. "
            )
            return None
        else:
            page_layouts, pdf_images = lp.load_pdf(pdf_path, load_images=True)
            document_md5sum = hashlib.md5(pdf_path.read_bytes()).hexdigest()

        # FIXME: handle EmptyFileError here using _pdf_num_pages
        model = _get_detectron_model(model, device)
        if ocr_agent == "tesseract":
            ocr_agent = lp.TesseractAgent()
        elif ocr_agent == "gcv":
            ocr_agent = lp.GCVAgent()

        num_pages = len(pdf_images)

        all_pages_metadata = []
        all_text_blocks = []

        logging.info(f"Iterating through pages for -  {input_task.document_id}")

        for page_idx, image in tqdm(
            enumerate(pdf_images), total=num_pages, desc=pdf_path.name
        ):
            page_dimensions = (
                page_layouts[page_idx].page_data["width"],
                page_layouts[page_idx].page_data["height"],
            )
            page_metadata = PDFPageMetadata(
                dimensions=page_dimensions,
                page_number=page_idx,
            )

            # If running in visual debug mode and the pdf is large, randomly select pages to save images for to avoid excessive redundancy
            # and processing time
            if debug:
                select_page = select_page_at_random(num_pages)
                if not select_page:
                    continue
            # Maybe we should always pass a layout object into the PageParser class.
            layout_disambiguator = LayoutDisambiguator(
                image, model, model_threshold_restrictive
            )
            initial_layout = layout_disambiguator.layout
            if len(initial_layout) == 0:
                logging.info(f"No layout found for page {page_idx}.")
                all_pages_metadata.append(page_metadata)
                continue
            disambiguated_layout = layout_disambiguator.disambiguate_layout()
            postprocessor = PostProcessor(disambiguated_layout)
            postprocessor.postprocess()
            ocr_blocks = postprocessor.ocr_blocks
            if len(ocr_blocks) == 0:
                logging.info(f"No text found for page {page_idx}.")
                all_pages_metadata.append(page_metadata)
                continue
            ocr_processor = OCRProcessor(
                image=np.array(image),
                page_number=page_idx,
                layout=ocr_blocks,
                ocr_agent=ocr_agent,
            )
            page_text_blocks, page_layout_blocks = ocr_processor.process_layout()
            # If running in visual debug mode, save images of the final layout to check how the model is performing.
            if debug:
                doc_name = input_task.document_name
                page_number = page_idx + 1
                image_output_path = (
                    Path(output_dir) / "debug" / f"{doc_name}_{page_number}.png"
                )

                page_layout = lp.Layout(page_layout_blocks)
                lp.draw_box(
                    image,
                    page_layout,
                    show_element_type=True,
                    box_alpha=0.2,
                    color_map={
                        "Inferred from gaps": "red",
                        "Ambiguous": "green",
                        "Text": "orange",
                        "Title": "blue",
                        "List": "brown",
                    },
                ).save(image_output_path)
            all_text_blocks += page_text_blocks

            all_pages_metadata.append(page_metadata)

        logging.info(f"Processing {input_task.document_id}, setting parser_output...")

        document = ParserOutput(
            document_id=input_task.document_id,
            document_name=input_task.document_name,
            document_description=input_task.document_description,
            document_source_url=input_task.document_source_url,
            document_content_type=input_task.document_content_type,
            document_cdn_object=input_task.document_cdn_object,
            document_md5_sum=input_task.document_md5_sum,
            document_slug=input_task.document_slug,
            document_metadata=input_task.document_metadata,
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
    debug: bool,
    device: str = "cpu",
) -> None:
    """
    Run cli to extract semi-structured JSON from document-AI + OCR.

    Args:
        input_tasks: List of tasks for the parser to process.
        output_dir: The directory to write the parsed PDFs to.
        parallel: Whether to run parsing over multiple processes.
        debug: Whether to run in debug mode (puts images of resulting layouts in output_dir).
        device: The device to use for the document AI model.
    """
    time_start = time.time()
    # ignore warnings that pollute the logs.
    warnings.filterwarnings("ignore")

    # Create logger that prints to stdout.
    logging.basicConfig(level=logging.DEBUG)

    logging.info(
        f"Using {config.PDF_OCR_AGENT} OCR agent and {config.LAYOUTPARSER_MODEL} model."
    )
    if config.PDF_OCR_AGENT == "gcv":
        logging.warning(
            "THIS IS COSTING MONEY/CREDITS!!!! - BE CAREFUL WHEN TESTING. SWITCH TO TESSERACT (FREE) FOR TESTING."
        )

    logging.info("Iterating through files and parsing pdf content from pages.")

    file_parser = partial(
        parse_file,
        model=config.LAYOUTPARSER_MODEL,
        ocr_agent=config.PDF_OCR_AGENT,
        output_dir=output_dir,
        debug=debug,
        model_threshold_restrictive=config.LAYOUTPARSER_MODEL_THRESHOLD_RESTRICTIVE,
        device=device,
    )
    if parallel:
        cpu_count = multiprocessing.cpu_count() - 1
        logging.info(f"Running in parallel and setting max workers to - {cpu_count}.")
        with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count) as executor:
            future_to_task = {executor.submit(file_parser, task): task for task in input_tasks}
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    data = future.result()
                except Exception as exc:
                    logging.exception('%r generated an exception: %s' % (task, exc))
                else:
                    logging.info(f'Result for {task} is {data}')

    else:
        for task in input_tasks:
            logging.info("Running in series.")
            file_parser(task)

    logging.info("Finished parsing pdf content from all files.")
    time_end = time.time()
    logging.info(f"Time taken: {time_end - time_start} seconds.")
