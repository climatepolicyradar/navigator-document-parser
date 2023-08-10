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
import hashlib

import cloudpathlib.exceptions
import requests
from azure.ai.formrecognizer import AnalyzeResult
from azure.core.exceptions import ServiceRequestError
from cloudpathlib import CloudPath, S3Path
from cpr_data_access.parser_models import (
    ParserInput,
    ParserOutput,
    PDFData,
)
from azure_pdf_parser import AzureApiWrapper, azure_api_response_to_parser_output

from src.config import AZURE_PROCESSOR_KEY, AZURE_PROCESSOR_ENDPOINT

CDN_DOMAIN = os.environ["CDN_DOMAIN"]
DOCUMENT_BUCKET_PREFIX = os.getenv("DOCUMENT_BUCKET_PREFIX", "navigator")

_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.DEBUG)


def copy_input_to_output_pdf(
    task: ParserInput, output_path: Union[Path, CloudPath]
) -> None:
    """Copy the input file to the output to ensure that we don't drop documents.

    The file is copied at the time of processing rather than syncing the entire input
    directory so that if that parser fails and retries it will not think that all
    files have already been parsed. :param task: input task specifying the document to
    copy :param output_path: path to save the copied file
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
            "Failed to save downloaded file locally. Content-Type is not "
            "application/pdf.",
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


def calculate_pdf_md5sum(file_path: str) -> str:
    """Calculate the md5sum of a pdf file."""
    with open(file_path, "rb") as file:
        pdf_data = file.read()
        md5_hash = hashlib.md5(pdf_data).hexdigest()
    return md5_hash


def read_local_json_to_bytes(path_: str) -> bytes:
    """Read a local json file into bytes"""
    with open(path_, 'rb') as file:
        return file.read()


def parse_file(
    input_task: ParserInput,
    azure_client: AzureApiWrapper,
    output_dir: Union[Path, S3Path],
    redo: bool = False,
):
    """Parse an individual pdf file.

    Args: azure_client (AzureApiWrapper): Client for interacting with Azure's
    services. input_task (ParserInput): Class specifying location of the PDF and other
    data about the task. output_dir (Path): Path to the output directory. redo (bool):
    Whether to redo the parsing even if the output file already exists.
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
    if not output_path.exists():
        copy_input_to_output_pdf(input_task, output_path)

    existing_parser_output = ParserOutput.parse_raw(output_path.read_text())
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
                "PDF path is None for document as the document either couldn't be "
                "downloaded, isn't content-type pdf or the response status code is not "
                "200.",
                extra={
                    "props": {
                        "document_id": input_task.document_id,
                        "temporary_local_location": temp_output_dir,
                    }
                },
            )
            return None

        # TODO retry with large document method if default fails
        try:
            api_response: AnalyzeResult = azure_client.analyze_document_from_bytes(
                doc_bytes=read_local_json_to_bytes(str(pdf_path)),
            )
        except ServiceRequestError as e:
            _LOGGER.exception(
                "Failed to parse document with Azure API. This is most likely due to "
                "incorrect azure api credentials.",
                extra={
                    "props": {
                        "document_id": input_task.document_id,
                        "error_message": str(e.message),
                    }
                },
            )
            return None
        except Exception as e:
            _LOGGER.exception(
                "Failed to parse document with Azure API.",
                extra={
                    "props": {
                        "document_id": input_task.document_id,
                        "error_message": str(e),
                    }
                },
            )
            return None

        document: ParserOutput = azure_api_response_to_parser_output(
            parser_input=input_task,
            md5_sum=calculate_pdf_md5sum(str(pdf_path)),
            api_response=api_response,
        )

        _LOGGER.info(
            "Saving parser output document.",
            extra={
                "props": {
                    "document_id": input_task.document_id,
                }
            },
        )

        try:
            output_path.write_text(document.json(indent=4, ensure_ascii=False))
        except cloudpathlib.exceptions.OverwriteNewerCloudError as e:
            _LOGGER.error(
                "Attempted write to s3, received OverwriteNewerCloudError and "
                "therefore skipping.",
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

    Args: input_tasks: List of tasks for the parser to process. output_dir: The
    directory to write the parsed PDFs to. parallel: Whether to run parsing over
    multiple processes. debug: Whether to run in debug mode (puts images of resulting
    layouts in output_dir). redo: Whether to redo the parsing even if the output file
    already exists.
    """
    time_start = time.time()
    # ignore warnings that pollute the logs.
    warnings.filterwarnings("ignore")

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

    azure_client = AzureApiWrapper(
        key=AZURE_PROCESSOR_KEY,
        endpoint=AZURE_PROCESSOR_ENDPOINT
    )

    file_parser = partial(
        parse_file,
        azure_client=azure_client,
        output_dir=output_dir,
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
                file_parser(input_task=task)
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
