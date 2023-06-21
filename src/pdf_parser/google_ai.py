import google
from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1
from google.cloud import documentai  # type: ignore
from layoutparser.elements import Rectangle


class GoogleAIAPIWrapper:
    def __init__(
        self, project_id: str, location: str, processor_id: str, mime_type: str
    ) -> None:
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

    def extract_document_text(
        self, image_content: str
    ) -> google.cloud.documentai_v1.Document:
        raw_document = documentai.RawDocument(
            content=image_content, mime_type=self.mime_type
        )

        request = documentai.ProcessRequest(name=self.name, raw_document=raw_document)

        return self.client.process_document(request=request)


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
