import concurrent
from ctypes import Union
from typing import Optional, Tuple, List

import numpy as np
from layoutparser.elements import Layout, TextBlock
from layoutparser.ocr import TesseractAgent, GCVAgent

from src.base import PDFTextBlock


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
            text = gcv_response.full_text_annotation.text

            # We assume one language per text block here which seems reasonable, but may not always be true.
            try:
                language = (
                    gcv_response.full_text_annotation.pages[0]
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
    def _remove_empty_text_blocks(layout: Layout) -> Layout:
        """Remove text blocks with no text from the layout."""
        # Heuristic to get rid of blocks with no text or text that is too short.
        ixs_to_remove = []
        for ix, block in enumerate(layout):
            if len(block.text.split(" ")) < 3:
                ixs_to_remove.append(ix)
        return Layout([b for ix, b in enumerate(layout) if ix not in ixs_to_remove])

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
                future = executor.submit(self._perform_ocr, self.image, block)
                block_with_text, block_language = future.result()
                if block_with_text is None:
                    continue
                if block.type == "Ambiguous":
                    block_with_text.type = self._infer_block_type(block)
                # Heuristic to get rid of blocks with no text or text that is too short.
                if not self._is_block_valid(block_with_text):
                    continue

                text_block_id = f"p_{self.page_number}_b_{block_idx}"
                text_block = PDFTextBlock.from_layoutparser(
                    block_with_text, text_block_id, self.page_number
                )
                text_block.language = block_language

                text_blocks.append(text_block)
                text_layout.append(block_with_text)

        return text_blocks, text_layout