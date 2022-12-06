import io
from collections import defaultdict
from typing import List

from PIL.PpmImagePlugin import PpmImageFile
from google.cloud import vision
from google.cloud.vision import types
from google.cloud.vision_v1.types import BoundingPoly
from layoutparser import Layout, TextBlock, Rectangle
from pydantic import BaseModel as PydanticBaseModel
from shapely.geometry import Polygon

from src.pdf_parser.pdf_utils.disambiguate_layout import lp_coords_to_shapely_polygon


class BaseModel(PydanticBaseModel):
    """Base class for all models."""

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True


# Hierarchical data structure for representing document text.
class GoogleTextSegment(BaseModel):
    text: str
    coordinates: BoundingPoly


class GoogleBlock(BaseModel):
    coordinates: BoundingPoly
    text_blocks: List[GoogleTextSegment]


class GoogleLayout:
    """Call the Google OCR API on a PDF page and return a Layout object."""

    def __init__(self, image: PpmImageFile) -> None:
        """Initialize the GoogleLayout object."""
        self.image = image
        self.block_texts = None
        self.paragraph_texts = None
        self.full_blocks = List[GoogleTextSegment]

    # save image as temp file and re-read to create bytes object
    @property
    def image_bytes(self) -> bytes:
        """Return the image as a bytes object."""
        image_buffer = io.BytesIO()
        self.image.save(image_buffer, format="PNG")
        image_buffer.seek(0)
        return image_buffer.read()

    @staticmethod
    def _google_vertex_to_shapely(bound):
        return Polygon(
            [
                (bound.vertices[0].x, bound.vertices[0].y),
                (bound.vertices[1].x, bound.vertices[1].y),
                (bound.vertices[2].x, bound.vertices[2].y),
                (bound.vertices[3].x, bound.vertices[3].y),
            ]
        )

    def extract_google_layout(self):
        """Returns document bounds given an image.

        The Google OCR API returns blocks of paragraphs. Roughly, there are 3 cases worth considering:

        1. A block consists of a heading and a paragraph of text below it. In this case the block consists of 2
        paragraphs, the first being the heading and the second being the text.
        2. A block consists of a paragraph of
        text. In this case the block and the paragraph have the same coordinates.
        3. A block consists of multiple  paragraphs of text. For example, if we have a list of items,
        each item is a paragraph.

        Given this, and given that we want to cross-reference the returned layout with the one returned by
        layoutparser + heuristics, we store blocks and paragraphs separately as "GoogleBlocks". To see the utility of this,
        consider 2 cases:

        1. Layout parser classifies a heading and a paragraph of text below it as 2 separate blocks,
         one for a heading and another for the following paragraph. In this case, the best coordinate matches
          with LayoutParser will be on what google classifies as paragraphs.
        2. Layout parser detects a list of items as a single block but is unable to detect the individual items.
        In this case, the best matches will be on what google classifies as blocks. So we want to merge the text
        from the paragraphs into the blocks.

        Regarding point 2, Google appears to be able to detect the individual items in a list more reliably than
        layoutparser, a point we can explore in the future to get the best results by combining the two approaches.
        For future usability, we also store block objects with paragraphs as sub-objects.
        """
        content = self.image_bytes
        client = vision.ImageAnnotatorClient()
        image = types.Image(content=content)

        # TODO: Handle errors. Hit a 503.
        response = client.document_text_detection(image=image)
        document = response.full_text_annotation

        breaks = vision.enums.TextAnnotation.DetectedBreak.BreakType
        paragraphs = []
        lines = []
        blocks = []
        paragraph_text_segments = []
        block_text_segments = []
        for page in document.pages:
            for block in page.blocks:
                block_paras = []
                block_para_coords = []
                for paragraph in block.paragraphs:
                    para = ""
                    line = ""
                    for word in paragraph.words:
                        for symbol in word.symbols:
                            line += symbol.text
                            if symbol.property.detected_break.type == breaks.SPACE:
                                line += " "
                            if (
                                symbol.property.detected_break.type
                                == breaks.EOL_SURE_SPACE
                            ):
                                line += " "
                                lines.append(line)
                                para += line
                                line = ""
                            if symbol.property.detected_break.type == breaks.LINE_BREAK:
                                lines.append(line)
                                para += line
                                line = ""
                    # for every paragraph, create a text block.
                    paragraph_text_segments.append(
                        GoogleTextSegment(text=para, coordinates=paragraph.bounding_box)
                    )
                    paragraphs.append(para)
                    block_paras.append(para)
                    block_para_coords.append(paragraph.bounding_box)

                # for every block, create a text block
                block_all_text = "\n".join(block_paras)
                block_text_segments.append(
                    GoogleTextSegment(
                        coordinates=block.bounding_box, text=block_all_text
                    )
                )

                # For every block, create a block object (contains paragraph metadata).
                block_list = [
                    GoogleTextSegment(coordinates=b2, text=b)
                    for b, b2 in zip(block_paras, block_para_coords)
                ]
                google_block = GoogleBlock(
                    coordinates=block.bounding_box, text_blocks=block_list
                )
                blocks.append(google_block)

        # look for duplicates in block_texts and paragraph_texts and create a list of full blocks
        blocks_to_keep = []
        for block in block_text_segments:
            if block in paragraph_text_segments:
                continue
            else:
                blocks_to_keep.append(block)

        all_segments = blocks_to_keep + paragraph_text_segments

        self.block_texts = block_text_segments
        self.paragraph_texts = paragraph_text_segments

        self.full_blocks = all_segments


def combine_google_lp(
    image,
    google_layout: GoogleLayout,
    lp_layout: Layout,
    threshold: float = 0.9,
    top_exclude: float = 0.1,
    bottom_exclude: float = 0.1,
):
    """
    Combine google layout with layoutparser layout.

    - Replaces layoutparser objects with objects recognised by google API + their text if they overlap sufficiently.
    - Does not include google objects if they seem to be not part of the main text (e.g. headers, footers, etc.). We
    use top exclude and bottom exclude to ascertain this.


    Args:
        image: The image the layout is based on.
        google_layout: GoogleLayout object
        lp_layout: Layout object
        threshold: Threshold for overlap between google and layoutparser objects. If the overlap is larger than
            threshold, the layoutparser object is replaced by the google object.
        top_exclude: Exclude objects from the top of the page if they are above this fraction of the page.
        bottom_exclude: Exclude objects from the bottom of the page if they are below this fraction of the page.

    Returns:
        Layout object with google objects replacing layoutparser objects if they overlap sufficiently.
    """
    google_layout.extract_google_layout()
    all_blocks: List[GoogleTextSegment] = google_layout.full_blocks
    shapely_google = [
        google_layout._google_vertex_to_shapely(b.coordinates) for b in all_blocks
    ]
    shapely_layout = [lp_coords_to_shapely_polygon(b.coordinates) for b in lp_layout]
    # for every block in the google layout, find the fraction of the block that is covered by the layoutparser layout
    dd_intersection_over_union = defaultdict(list)
    for ix_goog, google_block in enumerate(shapely_google):
        for ix_lp, lp_block in enumerate(shapely_layout):
            # Find the intersection over union of the two blocks.
            intersection = google_block.intersection(lp_block).area
            union = google_block.union(lp_block).area
            dd_intersection_over_union[ix_goog].append(intersection / union)

    # If a google block is covered by a layoutparser block by more than 0.9, we can assume that google OCR has
    # identified the same block as layoutparser + heuristics and we can use the text from the google block.
    # Filter the default dict for these cases
    equivalent_block_mapping = {
        k: v.index(max(v))
        for k, v in dd_intersection_over_union.items()
        if max(v) > threshold
    }

    # New blocks to add to the layoutparser layout
    blocks_google_only = {
        k: v for k, v in dd_intersection_over_union.items() if max(v) == 0.0
    }

    # use the mapping to replace the text of the layoutparser block with the text of the google block
    for k, v in equivalent_block_mapping.items():
        google_coords = all_blocks[k].coordinates.vertices
        x_top_left, y_top_left = google_coords[0].x, google_coords[0].y
        x_bottom_right, y_bottom_right = google_coords[2].x, google_coords[2].y
        lp_layout[v].text = all_blocks[k].text
        # Reset the coordinates of the layoutparser block to the coordinates of the google block.
        lp_layout[v].block.x_1 = x_top_left
        lp_layout[v].block.y_1 = y_top_left
        lp_layout[v].block.x_2 = x_bottom_right
        lp_layout[v].block.y_2 = y_bottom_right

    # Add the blocks that are only in the google layout, but only if they are not too small or are too high
    # up/down on the page indicating that they are probably not part of the main text.
    for key, val in blocks_google_only.items():
        google_coords = all_blocks[key].coordinates.vertices
        x_top_left, y_top_left = google_coords[0].x, google_coords[0].y
        x_bottom_right, y_bottom_right = google_coords[2].x, google_coords[2].y
        if (
            y_top_left > image.height * bottom_exclude
            or y_bottom_right < image.height * top_exclude
        ):
            rect = Rectangle(
                x_1=x_top_left, y_1=y_top_left, x_2=x_bottom_right, y_2=y_bottom_right
            )
            lp_layout.append(
                TextBlock(
                    rect,
                    text=all_blocks[key].text,
                    type="GoogleTextBlock",
                    score=1.0,  # TODO: Default to 1.0 for now but check if google has a score.
                )
            )

    return lp_layout
