import io
from typing import Any, Optional

from PyPDF2 import PdfReader, PdfWriter
from google.api_core.client_options import ClientOptions
from google.cloud import documentai  # type: ignore
from google.cloud import documentai_v1
from layoutparser.elements import Rectangle
from pydantic import BaseModel


class PDFPage(BaseModel):
    """Represents a batch of pages from a PDF document."""

    page_number: int
    extracted_content: Any


class GoogleTextBlockContent(BaseModel):
    """Represents a text block from the Google AI API."""

    layout: Any
    text: Optional[str] = None
    coordinates: Optional[Any] = None


class GoogleAIAPIWrapper:
    """Wrapper for the Google AI API."""

    def __init__(
        self, project_id: str, location: str, processor_id: str, mime_type: str
    ) -> None:
        """Initializes the Google AI API client."""
        self.project_id = project_id
        self.location = location
        self.processor_id = processor_id
        self.mime_type = mime_type
        self.opts = ClientOptions(
            api_endpoint=f"{self.location}-documentai.googleapis.com"
        )
        self.client = documentai.DocumentProcessorServiceClient(
            client_options=self.opts
        )
        self.name = self.client.processor_path(
            self.project_id, self.location, self.processor_id
        )

    def call_ai_api(self, page_content: bytes) -> Any:
        """Calls the Google AI API to extract text from a PDF page."""
        raw_document = documentai.RawDocument(
            content=page_content, mime_type=self.mime_type
        )

        request = documentai.ProcessRequest(name=self.name, raw_document=raw_document)

        return self.client.process_document(request=request).document

    def extract_document_text(self, pdf_path: str) -> list[PDFPage]:
        """Extracts text from a PDF document using the Google AI API."""
        pdf = PdfReader(pdf_path)

        pages_dict = {}
        total_pages = len(pdf.pages)
        for page_num in range(total_pages):
            writer = PdfWriter()
            writer.add_page(pdf.pages[page_num])
            pdf_bytes = io.BytesIO()
            writer.write(pdf_bytes)
            pdf_bytes.seek(0)
            pages_dict[page_num + 1] = pdf_bytes.read()

        return [
            PDFPage(
                page_number=page_num,
                extracted_content=self.call_ai_api(page_bytes),
            )
            for page_num, page_bytes in pages_dict.items()
        ]


def get_google_ai_text_blocks(
    page: documentai_v1.Document.Page,
    document_text: str,
) -> list[GoogleTextBlockContent]:
    """Converts Google AI layout coordinates to a GoogleTextBlockContent.

    This object contains layoutparser coordinates, text and the layout itself.
    It is necessary to scale the coordinates to the size of the page as they are initially normalized.
    """
    page_vertices = page.layout.bounding_poly.vertices[2]

    page_block_content = []

    for paragraph in page.paragraphs:
        text = layout_to_text(layout=paragraph.layout, text=document_text)

        box_vertices = paragraph.layout.bounding_poly.normalized_vertices

        rectangle_scaled = Rectangle(
            x_1=box_vertices[0].x * page_vertices.x,
            y_1=box_vertices[0].y * page_vertices.y,
            x_2=box_vertices[2].x * page_vertices.x,
            y_2=box_vertices[2].y * page_vertices.y,
        )

        page_block_content.append(
            GoogleTextBlockContent(
                layout=paragraph.layout,
                text=text,
                coordinates=rectangle_scaled,
            )
        )

    return page_block_content


def layout_to_text(layout: documentai.Document.Page.Layout, text: str) -> str:
    """
    This function converts offsets to a string.

    Document AI identifies text in different parts of the document by their offsets in the entirety of the
    document's text. If a text segment spans several lines, it will be stored in different text segments.
    """
    response = ""
    for segment in layout.text_anchor.text_segments:
        start_index = int(segment.start_index)
        end_index = int(segment.end_index)
        response += text[start_index:end_index]
    return response
