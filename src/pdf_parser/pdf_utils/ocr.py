import concurrent.futures
import io
from collections import defaultdict
from typing import Optional, Tuple, List, Union

import numpy as np
from PIL.PpmImagePlugin import PpmImageFile
from google.cloud import vision
from google.cloud.vision import types
from google.cloud.vision_v1.types import BoundingPoly  # type: ignore
from google.protobuf.pyext._message import RepeatedCompositeContainer
from layoutparser import TextBlock, Rectangle, Layout  # type: ignore
from layoutparser.ocr import TesseractAgent, GCVAgent
from shapely.geometry import Polygon

from src.base import PDFTextBlock
from src.pdf_parser.pdf_utils.disambiguate_layout import lp_coords_to_shapely_polygon
from src.pdf_parser.pdf_utils.utils import BaseModel


# Hierarchical data structure for representing document text.
class GoogleTextSegment(BaseModel):
    """A segment of text from Google OCR."""

    text: str
    coordinates: BoundingPoly
    confidence: float
    language: Optional[str]


class GoogleBlock(BaseModel):
    """A fully structured block from google OCR. Can contain multiple segments."""

    coordinates: BoundingPoly
    text_blocks: List[GoogleTextSegment]


def image_bytes(image: PpmImageFile) -> bytes:
    """Return the image as a bytes object."""
    image_buffer = io.BytesIO()
    image.save(image_buffer, format="PNG")
    image_buffer.seek(0)
    return image_buffer.read()


def google_vertex_to_shapely(bound):
    return Polygon(
        [
            (bound.vertices[0].x, bound.vertices[0].y),
            (bound.vertices[1].x, bound.vertices[1].y),
            (bound.vertices[2].x, bound.vertices[2].y),
            (bound.vertices[3].x, bound.vertices[3].y),
        ]
    )


def extract_google_layout(
    image: PpmImageFile,
) -> Tuple[
    List[GoogleBlock],
    List[GoogleTextSegment],
    List[GoogleTextSegment],
    List[GoogleTextSegment],
]:
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

    Args:
        image: Image to extract document bounds from.

    Returns:
        List of GoogleBlocks, List of GoogleTextSegments, List of GoogleTextSegments, List of GoogleTextSegments
    """

    def _get_modal_string(string_list: List[str]) -> Optional[str]:
        if len(string_list) > 0:
            return max(set(block_languages), key=block_languages.count)
        else:
            return None

    content = image_bytes(image)
    client = vision.ImageAnnotatorClient()
    image = types.Image(content=content)  # type: ignore

    # TODO: Handle errors. Hit a 503.
    response = client.document_text_detection(image=image)  # type: ignore
    document = response.full_text_annotation

    breaks = vision.enums.TextAnnotation.DetectedBreak.BreakType
    lines = []
    fully_structured_blocks = []
    paragraph_text_segments = []
    block_text_segments = []
    for page in document.pages:
        for block in page.blocks:
            block_languages = []
            default_dict = defaultdict(list)
            for paragraph in block.paragraphs:
                default_dict["paragraph_confidences"].append(paragraph.confidence)
                para = ""
                line = ""
                para_languages = (
                    []
                )  # languages stored at word level. Detect then take mode.
                for word in paragraph.words:
                    try:
                        lang = word.property.detected_languages[0].language_code
                    except IndexError:
                        lang = None
                    para_languages.append(lang)
                    block_languages.append(lang)
                    default_dict["block_languages"].append(lang)
                    for symbol in word.symbols:
                        line += symbol.text
                        break_type = symbol.property.detected_break.type
                        # Add space to end of line if it's not a break.
                        if break_type in [breaks.SPACE, breaks.EOL_SURE_SPACE]:
                            line += " "
                        # Start new line in same paragraph if there is a line break or a sure space (i.e. large space)
                        if break_type in [breaks.LINE_BREAK, breaks.EOL_SURE_SPACE]:
                            lines.append(line)
                            para += line
                            line = ""
                # Detect language by selecting the modal language of the words in the paragraph.
                para_lang = _get_modal_string(para_languages)
                paragraph_text_segments.append(
                    GoogleTextSegment(
                        text=para,
                        coordinates=paragraph.bounding_box,
                        confidence=paragraph.confidence,
                        language=para_lang,
                    )
                )
                default_dict["block_paragraphs"].append(para)
                default_dict["block_paragraph_coords"].append(paragraph.bounding_box)

            # Detect language by selecting the modal language of the words in the block.
            block_lang = _get_modal_string(default_dict["block_languages"])
            # for every block, create a text block
            block_all_text = "\n".join(default_dict["block_paragraphs"])
            block_text_segments.append(
                GoogleTextSegment(
                    coordinates=block.bounding_box,
                    text=block_all_text,
                    confidence=block.confidence,
                    language=block_lang,
                )
            )

            # For every block, create a block object (contains paragraph metadata).
            block_list = [
                GoogleTextSegment(
                    coordinates=default_dict["block_paragraph_coords"][i],
                    text=default_dict["block_paragraphs"][i],
                    language=default_dict["block_languages"][i],
                    confidence=default_dict["paragraph_confidences"][i],
                )
                for i in range(len(default_dict["block_paragraphs"]))
            ]
            google_block = GoogleBlock(
                coordinates=block.bounding_box, text_blocks=block_list
            )
            fully_structured_blocks.append(google_block)

    # look for duplicates in block_texts and paragraph_texts and create a list of full blocks
    text_blocks_to_keep = []
    for block in block_text_segments:
        if block in paragraph_text_segments:
            continue
        else:
            text_blocks_to_keep.append(block)

    combined_text_segments = text_blocks_to_keep + paragraph_text_segments

    return (
        fully_structured_blocks,
        combined_text_segments,
        block_text_segments,
        paragraph_text_segments,
    )


def google_coords_to_lp_coords(
    google_coords: RepeatedCompositeContainer,
) -> Tuple[int, int, int, int]:
    """Converts Google OCR coordinates to LayoutParser coordinates.

    Args:
        google_coords: Google OCR coordinates.

    Returns:
        Tuple of (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = (
        google_coords[0].x,
        google_coords[0].y,
        google_coords[2].x,
        google_coords[2].y,
    )
    return x1, y1, x2, y2


def combine_google_lp(
    image,
    google_layout: List[GoogleTextSegment],
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
        google_layout: List of GoogleTextSegment objects inferred from the structure returned by the Google OCR API.
        lp_layout: Layout object
        threshold: Threshold for overlap between google and layoutparser objects. If the overlap is larger than
            threshold, the layoutparser object is replaced by the google object.
        top_exclude: Exclude objects from the top of the page if they are above this fraction of the page.
        bottom_exclude: Exclude objects from the bottom of the page if they are below this fraction of the page.

    Returns:
        Layout object with google objects replacing layoutparser objects if they overlap sufficiently.
    """
    shapely_google = [google_vertex_to_shapely(b.coordinates) for b in google_layout]
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
        google_coords = google_layout[k].coordinates.vertices
        (
            x_top_left,
            y_top_left,
            x_bottom_right,
            y_bottom_right,
        ) = google_coords_to_lp_coords(google_coords)
        lp_layout[v].text = google_layout[k].text
        # TODO: This is ugly. Should create a data type to make these changes more explicit/to not duplicate code
        lp_layout[v].language = google_layout[k].language
        lp_layout[v].confidence = google_layout[k].confidence
        # Reset the coordinates of the layoutparser block to the coordinates of the google block.
        lp_layout[v].block.x_1 = x_top_left
        lp_layout[v].block.y_1 = y_top_left
        lp_layout[v].block.x_2 = x_bottom_right
        lp_layout[v].block.y_2 = y_bottom_right

    # Add the blocks that are only in the google layout, but only if they are not too small or are too high
    # up/down on the page indicating that they are probably not part of the main text.
    for key, val in blocks_google_only.items():
        google_coords = google_layout[key].coordinates.vertices
        (
            x_top_left,
            y_top_left,
            x_bottom_right,
            y_bottom_right,
        ) = google_coords_to_lp_coords(google_coords)
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
                    text=google_layout[key].text,
                    type="Google Text Block",
                    score=google_layout[key].confidence,
                )
            )
            # TODO: This is ugly. Should create a data type to make these changes more explicit/to not duplicate code
            lp_layout[-1].language = google_layout[key].language

    return lp_layout


class OCRProcessor:
    """Helper class for parsing text from a layout computer vision model output.

    Attributes:
        image: The image to process.
        page_number: Page number, starting from 0.
        layout: The processed layout of the page.
        ocr_agent: The OCR agent to use for text extraction. (e.g. Tesseract or Google Cloud Vision)
    """

    def __init__(
        self,
        image: np.ndarray,
        page_number: int,
        layout: Layout,
        ocr_agent: Union[TesseractAgent, GCVAgent],
    ):
        self.image = image
        self.page_number = page_number
        self.layout = layout
        self.ocr_agent = ocr_agent

    @staticmethod
    def _infer_block_type(block):
        """Try to infer a block's type using heuristics (and possibly a model later down the line).

        Args:
            block: The block to infer the type of.
        """
        # TODO: Write heuristics to infer the block type. Lists are especially important here.
        #  We need linguistic heuristics to determine the block type so this is dependent on OCR
        #   being carried out upstream of this function.
        return block.type

    def _perform_ocr(
        self,
        image: np.ndarray,
        block: TextBlock,
        left_pad: int = 15,
        right_pad: int = 15,
        top_pad: int = 2,
        bottom_pad: int = 2,
    ) -> Tuple[TextBlock, Optional[str]]:
        """Perform OCR on a block of text.

        Args:
            image: The image to perform OCR on.
            block: The block to set the text of.
            left_pad: The number of pixels to pad the left side of the block.
            right_pad: The number of pixels to pad the right side of the block.
            top_pad: The number of pixels to pad the top of the block.
            bottom_pad: The number of pixels to pad the bottom of the block.

        Returns:
            TextBlock: text block with the text set.
            str: the language of the text or None if the OCR processor doesn't support language detection.
        """
        # TODO: THis won't work currently because the image isn't part of the class.
        # Pad to improve OCR accuracy as it's fairly tight.

        padded_block = block.pad(
            left=left_pad, right=right_pad, top=top_pad, bottom=bottom_pad
        )

        segment_image = padded_block.crop_image(image)

        # Perform OCR
        if isinstance(self.ocr_agent, TesseractAgent):
            language = None
            text = self.ocr_agent.detect(segment_image, return_only_text=True)
        elif isinstance(self.ocr_agent, GCVAgent):
            gcv_response = self.ocr_agent.detect(segment_image, return_response=True)
            text = gcv_response.full_text_annotation.text  # type: ignore

            # We assume one language per text block here which seems reasonable, but may not always be true.
            try:
                language = (
                    gcv_response.full_text_annotation.pages[0]  # type: ignore
                    .property.detected_languages[0]
                    .language_code
                )
            except IndexError:
                # No language was found in the GCV response
                language = None
        else:
            raise ValueError(
                "The OCR agent must be either a TesseractAgent or a GCVAgent."
            )

        # Save OCR result
        block_with_text = padded_block.set(text=text)  # type: ignore

        return block_with_text, language  # type: ignore

    @staticmethod
    def _is_block_valid(block: TextBlock) -> bool:
        """Check if a block is valid."""
        if block.text is None:
            return False
        if len(block.text) == 0:
            return False
        # Heuristic to get rid of blocks with no text or text that is too short.
        if block.type == "Inferred from gaps":
            if len(block.text.split(" ")) < 3:
                return False
        return True

    def process_layout(self) -> Tuple[List[PDFTextBlock], List[TextBlock]]:
        """Get text for blocks in the layout and return a `Page` with text, language id per text block

        :return: list of text blocks with text, language and text block IDs set + a list of text blocks in
        layoutparser's format (useful for visual debugging).
        """
        text_blocks, text_layout = [], []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for block_idx, block in enumerate(self.layout):
                text_block_id = f"p_{self.page_number}_b_{block_idx}"
                # Skip blocks that already have text (likely because we used google's structure detection)
                if block.text is not None:
                    block_with_text = block
                    block_language = block.language
                    if not self._is_block_valid(block):
                        continue
                else:
                    future = executor.submit(self._perform_ocr, self.image, block)
                    block_with_text, block_language = future.result()
                    if block_with_text is None:
                        continue
                    if block.type == "Ambiguous":
                        block_with_text.type = self._infer_block_type(block)
                    # Heuristic to get rid of blocks with no text or text that is too short.
                    if not self._is_block_valid(block_with_text):
                        continue

                text_block = PDFTextBlock.from_layoutparser(
                    block_with_text, text_block_id, self.page_number
                )
                text_block.language = block_language

                text_blocks.append(text_block)
                text_layout.append(block_with_text)

        return text_blocks, text_layout
