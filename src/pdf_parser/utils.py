import io
from io import BytesIO
from typing import Sequence

from azure.ai.formrecognizer import AnalyzeResult, DocumentParagraph, Point
from PyPDF2 import PdfReader, PdfWriter

from src.base import ParserOutput, PDFTextBlock, PDFData, \
    PDFPageMetadata, PDFPage, TableCell, BoundingRegion, PDFTableBlock, ParserInput


def polygon_to_coords(polygon: Sequence[Point]) -> list[tuple[float, float]]:
    """Converts a polygon (four x,y co-ordinates) to a list of co-ordinates (two x,y points).

    The origin of the co-ordinate system is the top left corner of the page.

    The points array is ordered as follows:
    - top left, top  right, bottom right, bottom left
    """

    if len(polygon) != 4:
        raise ValueError("Polygon must have exactly four points.")

    return [(vertex.x, vertex.y) for vertex in polygon]


def convert_to_text_block(
    paragraph_id: int, paragraph: DocumentParagraph
) -> PDFTextBlock:
    """Convert a DocumentParagraph to a PDFTextBlock."""
    return PDFTextBlock(
        coords=polygon_to_coords(paragraph.bounding_regions[0].polygon),
        # FIXME: The paragraph could be split across multiple pages, page_number only allows int
        page_number=paragraph.bounding_regions[0].page_number,
        text=[paragraph.content],
        text_block_id=str(paragraph_id),
        language=None,
        type=paragraph.role or "Ambiguous",
        type_confidence=1.0,
    )


def convert_to_parser_output(parser_input: ParserInput, md5sum: str, api_response: AnalyzeResult) -> ParserOutput:
    """Convert the API response AnalyzeResult object to a ParserOutput."""
    return (
        ParserOutput(
            document_id=parser_input.document_id,
            document_metadata=parser_input.document_metadata,
            document_name=parser_input.document_name,
            document_description=parser_input.document_description,
            document_source_url=parser_input.document_source_url,
            document_cdn_object=parser_input.document_cdn_object,
            document_content_type=parser_input.document_cdn_object,
            document_md5_sum=md5sum,
            document_slug=parser_input.document_slug,
            languages=None,
            translated=False,
            html_data=None,
            pdf_data=PDFData(
                # FIXME: Check that the units of the dimensions are correct (units are in inches)
                page_metadata=[
                    PDFPageMetadata(
                        page_number=page.page_number,
                        dimensions=(page.width, page.height),
                    )
                    for page in api_response.pages
                ],
                md5sum=md5sum,
                text_blocks=[
                    convert_to_text_block(paragraph_id=index, paragraph=paragraph)
                    for index, paragraph in enumerate(api_response.paragraphs)
                ],
                table_blocks=[
                    PDFTableBlock(
                        table_id=0,
                        row_count=table.row_count,
                        column_count=table.column_count,
                        cells=[
                            TableCell(
                                cell_type=cell.kind,
                                row_index=cell.row_index,
                                column_index=cell.column_index,
                                row_span=cell.row_span,
                                column_span=cell.column_span,
                                content=cell.content,
                                bounding_regions=[
                                    BoundingRegion(
                                        page_number=cell.bounding_regions[
                                            0
                                        ].page_number,
                                        polygon=cell.bounding_regions[0].polygon,
                                    )
                                ],
                            )
                            for cell in table.cells
                        ],
                    )
                    for index, table in enumerate(api_response.tables)
                ],
            ),
        )
        .detect_and_set_languages()
        .set_document_languages_from_text_blocks()
    )


def propagate_with_correct_page_number(page: PDFPage) -> PDFPage:
    """Propagate the page number to the paragraphs and tables."""
    for paragraph in page.extracted_content.paragraphs:
        paragraph.bounding_regions[0].page_number = page.page_number

    for table in page.extracted_content.tables:
        for cell in table.cells:
            for bounding_region in cell.bounding_regions:
                bounding_region.page_number = page.page_number

        for bounding_region in table.bounding_regions:
            bounding_region.page_number = page.page_number
    return page


def merge_responses(pages: Sequence[PDFPage]) -> AnalyzeResult:
    """Merge individual page responses from multiple API calls into one.

    Currently, merging is done by concatenating the paragraphs and tables from each page.
    """

    pages = [propagate_with_correct_page_number(page) for page in pages]

    page_merged = pages[0].extracted_content

    [
        page_merged.paragraphs.append(page.extracted_content.paragrahs)
        for page in pages[1:]
    ]

    [page_merged.tables.append(page.extracted_content.tables) for page in pages[1:]]

    return page_merged


def split_into_pages(document_bytes: BytesIO) -> dict[int, bytes]:
    """Split the API response into individual pages."""
    pdf = PdfReader(document_bytes)

    pages_dict = {}
    total_pages = len(pdf.pages)
    for page_num in range(total_pages):
        writer = PdfWriter()
        writer.add_page(pdf.pages[page_num])
        pdf_bytes = io.BytesIO()
        writer.write(pdf_bytes)
        pdf_bytes.seek(0)
        pages_dict[page_num + 1] = pdf_bytes.read()

    return pages_dict
