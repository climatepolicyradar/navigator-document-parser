import concurrent.futures
import hashlib
import logging
import multiprocessing
import os
import tempfile
import time
import warnings
from functools import partial
from pathlib import Path
from typing import List, Optional, Union, cast

import cloudpathlib.exceptions
import fitz
import numpy as np
import requests
from cloudpathlib import CloudPath, S3Path
from fitz.fitz import EmptyFileError
from layoutparser import load_pdf, Layout, draw_box  # type: ignore
from layoutparser.models import Detectron2LayoutModel
from layoutparser.ocr import TesseractAgent, GCVAgent
from tqdm import tqdm

from src import config  # noqa: E402
from src.base import (  # noqa: E402
    ParserInput,
    ParserOutput,
    PDFData,
    PDFPageMetadata,
)
from src.pdf_parser.pdf_utils.disambiguate_layout import (
    run_disambiguation_pipeline,
    unnest_boxes,
)
from src.pdf_parser.pdf_utils.ocr import (
    OCRProcessor,
    extract_google_layout,
    combine_google_lp,
)
from src.pdf_parser.pdf_utils.postprocess_layout import postprocessing_pipline

CDN_DOMAIN = os.environ["CDN_DOMAIN"]

_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.DEBUG)


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

        try:
            output_path.write_text(blank_output.json(indent=4, ensure_ascii=False))
            _LOGGER.info(
                "Blank output saved.",
                extra={
                    "props": {
                        "document_id": task.document_id,
                        "output_path": output_path,
                    }
                },
            )
        except Exception as e:
            _LOGGER.exception(
                "Failed to write to output path.",
                extra={
                    "props": {
                        "document_id": task.document_id,
                        "output_path": output_path,
                        "error_message": str(e),
                    }
                },
            )

    except Exception as e:
        _LOGGER.exception(
            "Failed to parse",
            extra={
                "props": {
                    "document_id": task.document_id,
                    "output_path": output_path,
                    "error_message": str(e),
                }
            },
        )


def download_pdf(
    parser_input: ParserInput,
    output_dir: Union[Path, str],
) -> Optional[Path]:
    """Get a PDF from a URL in a ParserInput object.

    :param: parser input
    :param: directory to save the PDF to
    :return: path to PDF file in output_dir
    """
    document_url = f"https://{CDN_DOMAIN}/{parser_input.document_cdn_object}"

    try:
        _LOGGER.info(
            "Downloading document from url to local directory.",
            extra={
                "props": {
                    "document_id": parser_input.document_id,
                    "document_url": document_url,
                    "output_directory": output_dir,
                }
            },
        )
        response = requests.get(document_url)
    except Exception as e:
        _LOGGER.exception(
            "Failed to download document from url.",
            extra={
                "props": {
                    "document_id": parser_input.document_id,
                    "document_url": document_url,
                    "error_message": str(e),
                }
            },
        )
        return None

    if response.status_code != 200:
        _LOGGER.exception(
            "Non 200 status code from response.",
            extra={
                "props": {
                    "document_id": parser_input.document_id,
                    "document_url": document_url,
                    "response_status_code": response.status_code,
                }
            },
        )

        return None
    # FIXME: Had to uncomment this to get it working on some PDFs, related to oclet/stream instead of application/pdf
    # elif response.headers["Content-Type"] != "application/pdf":
    #     print(response.headers["Content-Type"])
    #     _LOGGER.error(
    #         StandardErrorLog.parse_obj(
    #             {
    #                 "timestamp": datetime.now(),
    #                 "pipeline_stage": "Parser: Validate Content-Type of downloaded file.",
    #                 "status_code": f"{response.status_code}",
    #                 "content_type": f"{response.headers['Content-Type']}",
    #                 "error_type": "ContentTypeError",
    #                 "message": "Content-Type is not application/pdf.",
    #                 "document_in_process": str(parser_input.document_id),
    #             }
    #         )
    #     )

    # return None

    else:
        _LOGGER.info(
            "Saving downloaded file locally.",
            extra={
                "props": {
                    "document_id": parser_input.document_id,
                    "document_url": document_url,
                }
            },
        )
        output_path = Path(output_dir) / f"{parser_input.document_id}.pdf"

        with open(output_path, "wb") as f:
            f.write(response.content)
        return output_path


def select_page_at_random(num_pages: int, rng: float) -> bool:
    """Determine whether to include a page using a random number generator. Used for debugging.

    Args:
        num_pages: The number of pages in the PDF.
        rng: A random number between 0 and 1.

    Returns:
        The page number to include.
    """
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
    unnest_soft_margin: float,
    min_overlapping_pixels_horizontal: int,
    min_overlapping_pixels_vertical: int,
    disambiguation_combination_threshold: float,
    ocr_agent: Union[TesseractAgent, GCVAgent],
    debug: bool,
    output_dir: Union[Path, S3Path],
    combine_google_vision: bool,
    top_exclude_threshold: float,
    bottom_exclude_threshold: float,
    replace_threshold: float,
    redo: bool = False,
):
    """Parse an individual pdf file.

    Args:
        input_task (ParserInput): Class specifying location of the PDF and other data about the task.
        model (layoutparser.LayoutModel): Layout model to use for parsing.
        model_threshold_restrictive (float): Threshold to use for parsing.
        unnest_soft_margin (int): Soft margin to use for unnesting (i.e. we expand a block by n pixels before
            performing is_in checks)
        min_overlapping_pixels_horizontal (int): Minimum number of pixel overlaps before reducing size to
            avoid OCR conflicts.
        min_overlapping_pixels_vertical (int): Minimum number of pixel overlaps before reducing size to
            avoid OCR conflicts.
        disambiguation_combination_threshold (float): Threshold to use for disambiguation.
        debug (bool): Whether to save debug images.
        ocr_agent (Union[TesseractAgent, GCVAgent]): OCR agent to use for parsing.
        output_dir (Path): Path to the output directory.
        device (str): Device to use for parsing.
        redo (bool): Whether to redo the parsing even if the output file already exists.
        bottom_exclude_threshold (float): Percentage of page to ignore at the bottom of the page when adding blocks
            to the page from google (e.g. to ignore footers).
        top_exclude_threshold (float): Percentage of page to ignore at the top of the page when adding blocks to the
            page from google (e.g. to ignore headers).
        replace_threshold (float): Threshold for replacing blocks from google with blocks from the model. e.g.
            if a block from layoutparser is 95% covered by a block from google, as measured by intersection over
            union, then the block from layoutparser will be replaced by the block from google.
        combine_google_vision (bool): Whether to combine the results from google vision with the results from the model.
    """

    _LOGGER.info(
        "Running pdf parser on document.",
        extra={
            "props": {
                "document_id": input_task.document_id,
            }
        },
    )

    output_path = cast(Path, output_dir / f"{input_task.document_id}.json")
    if not output_path.exists():  # type: ignore
        copy_input_to_output_pdf(input_task, output_path)  # type: ignore

    existing_parser_output = ParserOutput.parse_raw(output_path.read_text())  # type: ignore
    # If no parsed pdf data exists, assume we've not run before
    existing_pdf_data_exists = (
        existing_parser_output.pdf_data is not None
        and existing_parser_output.pdf_data.text_blocks
    )
    should_run_parser = not existing_pdf_data_exists or redo
    if not should_run_parser:
        _LOGGER.info(
            "Skipping already parsed pdf.",
            extra={
                "props": {
                    "document_id": input_task.document_id,
                    "output_path": output_path,
                }
            },
        )
        return None

    with tempfile.TemporaryDirectory() as temp_output_dir:
        _LOGGER.info(f"Downloading pdf: {input_task.document_id}")
        pdf_path = download_pdf(input_task, temp_output_dir)
        _LOGGER.info(f"PDF path for: {input_task.document_id} - {pdf_path}")
        if pdf_path is None:
            _LOGGER.info(
                "PDF path is None for document as the document either couldn't be downloaded, isn't content-type pdf "
                "or the response status code is not 200.",
                extra={
                    "props": {
                        "document_id": input_task.document_id,
                        "temporary_local_location": temp_output_dir,
                    }
                },
            )
            return None
        else:
            page_layouts, pdf_images = load_pdf(pdf_path, load_images=True)  # type: ignore
            document_md5sum = hashlib.md5(pdf_path.read_bytes()).hexdigest()

        num_pages = len(pdf_images)

        random_numbers = np.random.RandomState(42).random(num_pages)

        all_pages_metadata = []
        all_text_blocks = []

        _LOGGER.info(
            "Iterating through pages.",
            extra={
                "props": {
                    "document_id": input_task.document_id,
                    "number_of_pages": num_pages,
                }
            },
        )

        for page_idx, image in tqdm(
            enumerate(pdf_images), total=num_pages, desc=pdf_path.name
        ):
            _LOGGER.info(
                "Processing page.",
                extra={
                    "props": {
                        "document_id": input_task.document_id,
                        "page_index": page_idx,
                    }
                },
            )
            page_dimensions = (
                page_layouts[page_idx].page_data["width"],
                page_layouts[page_idx].page_data["height"],
            )
            page_metadata = PDFPageMetadata(
                dimensions=page_dimensions,
                page_number=page_idx,
            )

            # If running in visual debug mode and the pdf is large, randomly select pages to save images for to avoid
            # excessive redundancy and processing time
            if debug:
                rng = random_numbers[page_idx]
                select_page = select_page_at_random(num_pages, rng)
                if not select_page:
                    continue
            # Maybe we should always pass a layout object into the PageParser class.
            _LOGGER.info(f"Running layout_disambiguator for page {page_idx}")
            layout_disambiguated = run_disambiguation_pipeline(
                image,
                model,
                restrictive_model_threshold=model_threshold_restrictive,
                unnest_soft_margin=unnest_soft_margin,  # type: ignore
                min_overlapping_pixels_horizontal=min_overlapping_pixels_horizontal,
                min_overlapping_pixels_vertical=min_overlapping_pixels_vertical,
                combination_threshold=disambiguation_combination_threshold,
            )
            if len(layout_disambiguated) == 0:
                _LOGGER.info(
                    f"The layoutparser model has found no layout elements of any type for page {page_idx}. Continuing to next page."
                )
                all_pages_metadata.append(page_metadata)
                continue
            _LOGGER.info(f"Running google document structure OCR for page {page_idx}")
            if combine_google_vision:
                # Add a step to use google vision instead of lists
                layout_disambiguated = Layout(
                    [b for b in layout_disambiguated if b.type != "List"]
                )
                google_layout = extract_google_layout(image)[1]
                # Combine the Google text blocks with the layoutparser layout.
                postprocessed_layout = combine_google_lp(
                    image,
                    google_layout,
                    layout_disambiguated,
                    threshold=replace_threshold,
                    top_exclude=top_exclude_threshold,
                    bottom_exclude=bottom_exclude_threshold,
                )
                # unnest the layout again because the google layout may have nested elements. Hack.
                postprocessed_layout = unnest_boxes(
                    postprocessed_layout, unnest_soft_margin=unnest_soft_margin
                )
            else:
                postprocessed_layout = postprocessing_pipline(
                    layout_disambiguated, page_dimensions[1]
                )
            ocr_blocks = Layout(
                [
                    b
                    for b in postprocessed_layout
                    if b.type
                    in [
                        "Google Text Block",
                        "Text",
                        "List",
                        "Title",
                        "Ambiguous",
                        "Inferred from gaps",
                    ]
                ]
            )
            if len(ocr_blocks) == 0:
                _LOGGER.info(f"No text found for page {page_idx}.")
                all_pages_metadata.append(page_metadata)
                continue
            ocr_processor = OCRProcessor(
                np.array(image), page_idx, postprocessed_layout, ocr_agent
            )
            _LOGGER.info(
                f"Running ocr at block level for unaccounted for blocks for page {page_idx}"
            )
            page_text_blocks, page_layout_blocks = ocr_processor.process_layout()

            # If running in visual debug mode, save images of the final layout to check how the model is performing.
            if debug:
                doc_name = input_task.document_name
                page_number = page_idx + 1
                image_output_path = (
                    Path(output_dir) / "debug" / f"{doc_name}_{page_number}.png"
                )
                page_layout = Layout(
                    [
                        b
                        for b in postprocessed_layout
                        if b.type
                        in [
                            "Text",
                            "List",
                            "Title",
                            "Ambiguous",
                            "Inferred from gaps",
                            "Google Text Block",
                        ]
                    ]
                )
                draw_box(
                    image,
                    page_layout,
                    show_element_type=True,
                    box_alpha=0.1,
                    color_map={
                        "Inferred from gaps": "red",
                        "Ambiguous": "green",
                        "Text": "orange",
                        "Title": "blue",
                        "List": "brown",
                        "Google Text Block": "purple",
                    },
                ).save(image_output_path)
            all_text_blocks += page_text_blocks

            all_pages_metadata.append(page_metadata)

        _LOGGER.info(
            "Setting parser output for document.",
            extra={
                "props": {
                    "document_id": input_task.document_id,
                }
            },
        )

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

        try:
            output_path.write_text(document.json(indent=4, ensure_ascii=False))
        except cloudpathlib.exceptions.OverwriteNewerCloudError as e:
            _LOGGER.error(
                "Attempted write to s3, received OverwriteNewerCloudError and therefore skipping.",
                extra={
                    "props": {
                        "document_id": input_task.document_id,
                        "output_path": output_path,
                        "error_message": str(e),
                    }
                },
            )

        _LOGGER.info(
            "Saved document.",
            extra={
                "props": {
                    "document_id": input_task.document_id,
                    "output_path": output_path.name,
                    "output_directory": output_dir,
                }
            },
        )

        os.remove(pdf_path)
        _LOGGER.info(
            "Removed downloaded document.",
            extra={
                "props": {
                    "document_id": input_task.document_id,
                    "local_document_path": pdf_path,
                }
            },
        )


def _pdf_num_pages(file: str):
    """Get the number of pages in a pdf file."""
    try:
        return fitz.open(file).page_count  # type: ignore
    except EmptyFileError:
        return 0


# TODO: We may want to make this an option, but for now just use Detectron by default as we are unlikely
#  to change this unless we start labelling by ourselves.
def _get_detectron_model(model: str, device: str) -> Detectron2LayoutModel:
    return Detectron2LayoutModel(
        config_path=f"lp://PubLayNet/{model}",  # In model catalog,
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
        device=device,
    )


def get_model(
    model_name: str,
    ocr_agent_type_name: str,
    device: str,
) -> tuple[Detectron2LayoutModel, Union[TesseractAgent, GCVAgent]]:
    """Get the model for the parser."""
    _LOGGER.info(
        "Model Configuration",
        extra={
            "props": {
                "model": model_name,
                "ocr_agent": ocr_agent_type_name,
                "device": device,
            }
        },
    )
    if config.PDF_OCR_AGENT == "gcv":
        _LOGGER.warning(
            "THIS IS COSTING MONEY/CREDITS!!!! - BE CAREFUL WHEN TESTING. SWITCH TO TESSERACT (FREE) FOR TESTING."
        )

    # FIXME: handle EmptyFileError here using _pdf_num_pages
    model = _get_detectron_model(model_name, device)
    if ocr_agent_type_name == "tesseract":
        ocr_agent = TesseractAgent()
    elif ocr_agent_type_name == "gcv":
        ocr_agent = GCVAgent()
    else:
        raise RuntimeError(f"Uknown OCR agent type: '{ocr_agent_type_name}'")

    return model, ocr_agent


def run_pdf_parser(
    input_tasks: List[ParserInput],
    output_dir: Union[Path, S3Path],
    parallel: bool,
    debug: bool,
    use_google_document_ai: bool,
    device: str = "cpu",
    redo: bool = False,
) -> None:
    """
    Run cli to extract semi-structured JSON from document-AI + OCR.

    Args:
        input_tasks: List of tasks for the parser to process.
        output_dir: The directory to write the parsed PDFs to.
        parallel: Whether to run parsing over multiple processes.
        debug: Whether to run in debug mode (puts images of resulting layouts in output_dir).
        device: The device to use for the document AI model.
        redo: Whether to redo the parsing even if the output file already exists.
        use_google_document_ai: Whether to use Google Document AI to help parse the PDFs.
    """
    time_start = time.time()
    # ignore warnings that pollute the logs.
    warnings.filterwarnings("ignore")

    model, ocr_agent = get_model(
        model_name=config.LAYOUTPARSER_MODEL,
        ocr_agent_type_name=config.PDF_OCR_AGENT,
        device=device,
    )

    _LOGGER.info(
        "Iterating through files and parsing pdf content from pages.",
        extra={
            "props": {
                "parallel": parallel,
                "debug": debug,
                "redo": redo,
                "number_of_tasks": len(input_tasks),
            },
        },
    )
    file_parser = partial(
        parse_file,
        model=model,
        combine_google_vision=use_google_document_ai,
        ocr_agent=ocr_agent,
        output_dir=output_dir,
        debug=debug,
        model_threshold_restrictive=config.LAYOUTPARSER_MODEL_THRESHOLD_RESTRICTIVE,
        unnest_soft_margin=config.LAYOUTPARSER_UNNEST_SOFT_MARGIN,
        disambiguation_combination_threshold=config.LAYOUTPARSER_DISAMBIGUATION_COMBINATION_THRESHOLD,
        min_overlapping_pixels_vertical=config.LAYOUTPARSER_MIN_OVERLAPPING_PIXELS_VERTICAL,
        min_overlapping_pixels_horizontal=config.LAYOUTPARSER_MIN_OVERLAPPING_PIXELS_HORIZONTAL,
        top_exclude_threshold=config.LAYOUTPARSER_TOP_EXCLUDE_THRESHOLD,
        bottom_exclude_threshold=config.LAYOUTPARSER_BOTTOM_EXCLUDE_THRESHOLD,
        replace_threshold=config.LAYOUTPARSER_REPLACE_THRESHOLD,
        redo=redo,
    )
    if parallel:
        cpu_count = min(3, multiprocessing.cpu_count() - 1)
        _LOGGER.info(
            "Running in parallel and setting max workers.",
            extra={"props": {"max_workers": cpu_count}},
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count) as executor:
            future_to_task = {
                executor.submit(file_parser, task): task for task in input_tasks
            }
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    data = future.result()  # noqa: F841
                except Exception as e:
                    _LOGGER.exception(
                        "Document failed to generate a result.",
                        extra={
                            "props": {
                                "document_id": task.document_id,
                                "error_message": str(e),
                            }
                        },
                    )
                else:
                    _LOGGER.info(
                        "Document successful parsed by pdf parser.",
                        extra={
                            "props": {
                                "document_id": task.document_id,
                            }
                        },
                    )

    else:
        for task in input_tasks:
            _LOGGER.info("Running in series.")
            try:
                file_parser(task)
            except Exception as e:
                _LOGGER.exception(
                    "Failed to successfully parse PDF due to a raised exception",
                    extra={
                        "props": {
                            "document_id": task.document_id,
                            "error_message": str(e),
                        }
                    },
                )

    time_end = time.time()
    _LOGGER.info(
        "PDF parsing complete for all files.",
        extra={
            "props": {
                "time_taken": time_end - time_start,
                "start_time": time_start,
                "end_time": time_end,
            }
        },
    )
