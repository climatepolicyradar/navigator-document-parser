import io
from typing import Any

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


def get_google_ai_layout_coords(
    page: documentai_v1.Document.Page,
) -> list[Rectangle]:
    """Converts Google AI layout coordinates to layoutparser coordinates.

    It is also necessary to scale the coordinates to the size of the page as they are initially normalized.
    """
    page_vertices = page.layout.bounding_poly.vertices[2]

    google_ai_blocks = [
        block.layout.bounding_poly.normalized_vertices for block in page.blocks
    ]

    google_ai_coords = [
        Rectangle(x_1=block[0].x, y_1=block[0].y, x_2=block[2].x, y_2=block[2].y)
        for block in google_ai_blocks
    ]

    google_ai_coords_scaled = [
        Rectangle(
            x_1=block.x_1 * page_vertices.x,
            y_1=block.y_1 * page_vertices.y,
            x_2=block.x_2 * page_vertices.x,
            y_2=block.y_2 * page_vertices.y,
        )
        for block in google_ai_coords
    ]

    return google_ai_coords_scaled
