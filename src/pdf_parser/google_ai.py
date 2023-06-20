import google
from google.api_core.client_options import ClientOptions
from google.cloud import documentai  # type: ignore


class GoogleAIAPIWrapper:
    def __init__(self, project_id: str, location: str, processor_id: str, mime_type: str) -> None:
        self.project_id = project_id
        self.location = location
        self.processor_id = processor_id
        self.mime_type = mime_type
        self.opts = ClientOptions(api_endpoint=f"{self.location}-documentai.googleapis.com")
        self.client = documentai.DocumentProcessorServiceClient(client_options=self.opts)
        self.name = self.client.processor_path(self.project_id, self.location, self.processor_id)

    def extract_document_text(self, image_content: str) -> google.cloud.documentai_v1.Document:
        raw_document = documentai.RawDocument(content=image_content, mime_type=self.mime_type)

        request = documentai.ProcessRequest(name=self.name, raw_document=raw_document)

        return self.client.process_document(request=request)
