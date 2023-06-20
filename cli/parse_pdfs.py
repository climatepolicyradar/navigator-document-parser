import concurrent.futures
import logging
import multiprocessing
import os
import tempfile
import time
import warnings
from functools import partial
from pathlib import Path
from typing import List, Optional, Union

import cloudpathlib.exceptions
import requests
from cloudpathlib import CloudPath, S3Path

from src.base import (
    ParserInput,
    ParserOutput,
    PDFData,
)

from src.config import PROJECT_ID, LOCATION, PROCESSOR_ID, MIME_TYPE
from src.pdf_parser.combine import assign_block_type
from src.pdf_parser.google_ai import GoogleAIAPIWrapper
from src.pdf_parser.layout import LayoutParserWrapper

CDN_DOMAIN = os.environ["CDN_DOMAIN"]
DOCUMENT_BUCKET_PREFIX = os.getenv("DOCUMENT_BUCKET_PREFIX", "navigator")

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
                        "output_path": str(output_path),
                    }
                },
            )
        except Exception as e:
            _LOGGER.exception(
                "Failed to write to output path.",
                extra={
                    "props": {
                        "document_id": task.document_id,
                        "output_path": str(output_path),
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
                    "output_path": str(output_path),
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
    document_url = os.path.join(
        *[
            "https://",
            CDN_DOMAIN,
            DOCUMENT_BUCKET_PREFIX,
            parser_input.document_cdn_object,
        ]
    )

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
    elif response.headers["Content-Type"] != "application/pdf":
        _LOGGER.exception(
            "Failed to save downloaded file locally. Content-Type is not application/pdf.",
            extra={
                "props": {
                    "document_id": parser_input.document_id,
                    "document_url": document_url,
                    "response_status_code": response.status_code,
                }
            },
        )
        return None
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


def parse_file(
    google_ai_client: GoogleAIAPIWrapper,
    lp_obj: LayoutParserWrapper,
    input_task: ParserInput,
    output_dir: Union[Path, S3Path],
    redo: bool = False,
):
    """Parse an individual pdf file.

    Args:
        google_ai_client (GoogleAIAPIWrapper): Client for interacting with Google's AI services.
        input_task (ParserInput): Class specifying location of the PDF and other data about the task.
        output_dir (Path): Path to the output directory.
        redo (bool): Whether to redo the parsing even if the output file already exists.
    """

    _LOGGER.info(
        "Running pdf parser on document.",
        extra={
            "props": {
                "document_id": input_task.document_id,
            }
        },
    )

    output_path = output_dir / f"{input_task.document_id}.json"
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
                    "output_path": str(output_path),
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

        # TODO FROM HERE

        with open(pdf_path, "rb") as document:
            document_content = document.read()
        googled_parsed_document = google_ai_client.extract_document_text(document_content)

        input_task, all_pages_metadata, document_md5_sum, all_text_blocks = assign_block_type(googled_parsed_document, lp_obj)



        # TODO TO HERE
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
                        "output_path": str(output_path),
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
                    "output_directory": str(output_dir),
                }
            },
        )

        os.remove(pdf_path)
        _LOGGER.info(
            "Removed downloaded document.",
            extra={
                "props": {
                    "document_id": input_task.document_id,
                    "local_document_path": str(pdf_path),
                }
            },
        )


def run_pdf_parser(
    input_tasks: List[ParserInput],
    output_dir: Union[Path, S3Path],
    parallel: bool,
    debug: bool,
    redo: bool = False,
) -> None:
    """
    Run cli to extract semi-structured JSON from document-AI + OCR.

    Args:
        input_tasks: List of tasks for the parser to process.
        output_dir: The directory to write the parsed PDFs to.
        parallel: Whether to run parsing over multiple processes.
        debug: Whether to run in debug mode (puts images of resulting layouts in output_dir).
        redo: Whether to redo the parsing even if the output file already exists.
    """
    time_start = time.time()
    # ignore warnings that pollute the logs.
    warnings.filterwarnings("ignore")

    google_ai_client = GoogleAIAPIWrapper(PROJECT_ID, LOCATION, PROCESSOR_ID, MIME_TYPE)
    lp_obj = LayoutParserWrapper()

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
        output_dir=output_dir,
        redo=redo,
        google_ai_client=google_ai_client,
        lp_obj=lp_obj,
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
