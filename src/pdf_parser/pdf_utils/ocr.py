import concurrent.futures
import io
from collections import defaultdict
from typing import Optional, Tuple, List, Union, Dict

import numpy as np
from tenacity import retry, wait_fixed
from PIL.PpmImagePlugin import PpmImageFile
from google.cloud import vision
from google.cloud.vision import types
from google.protobuf.pyext._message import RepeatedCompositeContainer
from layoutparser import TextBlock, Rectangle, Layout  # type: ignore
from layoutparser.ocr import TesseractAgent, GCVAgent
from shapely.geometry import Polygon
import logging

from src.base import PDFTextBlock, GoogleBlock, GoogleTextSegment
from src.pdf_parser.pdf_utils.disambiguator.unexplained import (
    lp_coords_to_shapely_polygon,
)

_LOGGER = logging.getLogger(__name__)


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


def get_modal_string(
    language_list: List[str], block_languages: List[str]
) -> Optional[str]:
    """Get the modal language from a list of languages.

    Args:
        language_list: List of languages to choose from.
        block_languages: List of languages that the block is in.

    Returns:
        The modal language.

    """
    if len(language_list) > 0:
        return max(set(block_languages), key=block_languages.count)
    else:
        return None


@retry(wait=wait_fixed(60))
def get_text_annotation(image_):
    """
    Get the text annotation from the Google OCR API.

    Retry once after 1 min if the API returns an error.
    """
    _LOGGER.info("Getting text annotation function called.")
    client = vision.ImageAnnotatorClient()
    response = client.document_text_detection(image=image_)
    if response.error.code == 503:
        raise Exception("Google OCR API returned 503. Try again later.")
    return response


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
    _LOGGER.info("Extracting Google layout from image.")

    content = image_bytes(image)
    image = types.Image(content=content)  # type: ignore

    # # TODO: Handle errors. Hit a 503.
    _LOGGER.info("Getting text annotation from Google OCR API.")
    response = get_text_annotation(image)
    _LOGGER.info("Got text annotation from Google OCR API.")
    document = response.full_text_annotation

    breaks = vision.enums.TextAnnotation.DetectedBreak.BreakType
    lines = []
    fully_structured_blocks = []
    paragraph_text_segments = []
    block_text_segments = []
    _LOGGER.info(
        "Iterating over Google OCR response.",
        extra={
            "props": {
                "total_num_pages": len(document.pages),
            },
        },
    )
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
                para_lang = get_modal_string(para_languages, block_languages)
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
            block_lang = get_modal_string(
                default_dict["block_languages"], block_languages
            )
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
    text_blocks_to_keep = [
        block for block in block_text_segments if block not in paragraph_text_segments
    ]

    combined_text_segments = text_blocks_to_keep + paragraph_text_segments
    _LOGGER.info(
        "Finished iterating over Google OCR response.",
        extra={
            "props": {
                "fully_structured_blocks": len(fully_structured_blocks),
                "combined_text_segments_length": len(combined_text_segments),
                "text_blocks_to_keep_length": len(text_blocks_to_keep),
                "paragraph_text_segments_length": len(paragraph_text_segments),
                "block_text_segments_length": len(block_text_segments),
                "paragragh_text_segments_length": len(paragraph_text_segments),
            }
        },
    )

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


def calculate_intersection_over_unions(
    shapely_google: List, shapely_layout: List
) -> defaultdict[int, list[float]]:
    """
    Calculate intersection over union for every block in the Google layout and LayoutParser layout.

    Args:
        shapely_google: A list of Shapely objects representing the blocks in the Google layout.
        shapely_layout: A list of Shapely objects representing the blocks in the LayoutParser layout.

    Returns:
        A dictionary that contains lists of intersection over unions for every google block with all lp blocks.
    """
    dd_intersection_over_union = defaultdict(list)
    for ix_goog, google_block in enumerate(shapely_google):
        for ix_lp, lp_block in enumerate(shapely_layout):
            # Find the intersection over union of the two blocks.
            intersection = google_block.intersection(lp_block).area
            union = google_block.union(lp_block).area
            dd_intersection_over_union[ix_goog].append(intersection / union)
    return dd_intersection_over_union


def find_equivalent_block_mapping(
    dd_intersection_over_union: Dict[int, List[float]], threshold: float
) -> Dict[int, int]:
    """
    Finds the equivalent block mapping between the Google layout and LayoutParser layout.

    Args:
        dd_intersection_over_union: A dictionary that contains the intersection over union for each block in the Google
            layout and LayoutParser layout.
        threshold: Threshold for overlap between google and layoutparser objects. If the overlap is larger than
            threshold, the layoutparser object is replaced by the google object.

    Returns:
        A dictionary that maps the index of each block in the Google layout to the index of the most overlapping block
        in the LayoutParser layout, for blocks with an intersection over union greater than the threshold.
    """
    equivalent_block_mapping = {
        k: v.index(max(v))
        for k, v in dd_intersection_over_union.items()
        if max(v) > threshold
    }
    return equivalent_block_mapping


def replace_block_text(
    equivalent_block_mapping: Dict[int, int], lp_layout: List, google_layout: List
) -> List:
    """
    Replaces the text of the LayoutParser blocks with the text of the Google blocks.

    Args:
        equivalent_block_mapping: A dictionary that maps the index of each block in the Google layout to the index of the
            most overlapping block in the LayoutParser layout, for blocks with an intersection over union greater than a threshold
            defined upstream.
        lp_layout: A list of LayoutParser blocks.
        google_layout: A list of Google blocks.

    Returns:
        A list of LayoutParser blocks with the text of the Google blocks.
    """
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
    return lp_layout


def add_google_specific_blocks(
    image,
    blocks_google_only: dict,
    google_layout: List[GoogleTextSegment],
    lp_layout: Layout,
    top_exclude: float,
    bottom_exclude: float,
) -> Layout:
    """
    Adds the Google blocks that do not overlap with any LayoutParser blocks to the LayoutParser layout.

    Args:
        image: The image that the LayoutParser layout was created from.
        blocks_google_only: A list of Google blocks that do not overlap with any LayoutParser blocks.
        google_layout: A list of Google blocks.
        lp_layout: A list of LayoutParser blocks.
        top_exclude: The number of pixels to exclude from the top of the image.
        bottom_exclude: The number of pixels to exclude from the bottom of the image.

    Returns:
        A list of LayoutParser blocks with the text of the Google blocks.
    """
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
        Layout object with google objects replacing layoutparser objects if they overlap sufficiently, plus any google
        specific blocks.
    """
    _LOGGER.debug(
        "Combining Google and LayoutParser layouts function called.",
        extra={
            "google_layout_length": len(google_layout),
            "lp_layout_length": len(lp_layout),
        },
    )
    _LOGGER.debug("Length of google layout: %s", len(google_layout))
    _LOGGER.debug("Length of layoutparser layout: %s", len(lp_layout))

    shapely_google = [google_vertex_to_shapely(b.coordinates) for b in google_layout]
    _LOGGER.debug(
        "Google layout converted to shapely objects.",
        extra={"props": {"shapely_google_length": len(shapely_google)}},
    )

    shapely_layout = [lp_coords_to_shapely_polygon(b.coordinates) for b in lp_layout]
    _LOGGER.debug(
        "LayoutParser layout converted to shapely objects.",
        extra={"props": {"shapely_layout_length": len(shapely_layout)}},
    )

    dd_intersection_over_union = calculate_intersection_over_unions(
        shapely_google, shapely_layout
    )
    _LOGGER.debug(
        "Intersection over unions calculated.",
        extra={
            "props": {
                "dd_intersection_over_union_length": len(dd_intersection_over_union)
            }
        },
    )

    equivalent_block_mapping = find_equivalent_block_mapping(
        dd_intersection_over_union, threshold
    )
    _LOGGER.debug(
        "Equivalent block mapping found.",
        extra={
            "props": {"equivalent_block_mapping_length": len(equivalent_block_mapping)}
        },
    )

    # New blocks to add to the layoutparser layout
    blocks_google_only = {
        k: v for k, v in dd_intersection_over_union.items() if max(v) == 0.0
    }
    _LOGGER.debug(
        "Google only blocks found.",
        extra={"props": {"blocks_google_only_length": len(blocks_google_only)}},
    )

    # use the mapping to replace the text of the layoutparser block with the text of the google block
    lp_layout = replace_block_text(equivalent_block_mapping, lp_layout, google_layout)
    _LOGGER.debug(
        "LayoutParser blocks replaced with Google blocks.",
        extra={"props": {"lp_layout_length": len(lp_layout)}},
    )

    lp_layout_with_google = add_google_specific_blocks(
        image, blocks_google_only, google_layout, lp_layout, top_exclude, bottom_exclude
    )
    _LOGGER.debug(
        "Google specific blocks added to LayoutParser layout.",
        extra={"props": {"lp_layout_with_google_length": len(lp_layout_with_google)}},
    )

    return lp_layout_with_google


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
