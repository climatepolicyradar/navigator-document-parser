import concurrent.futures
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Tuple, Union, Optional

import layoutparser as lp
import numpy as np
import pandas as pd
from layoutparser import ocr
from shapely.geometry import LineString
from shapely.geometry.polygon import Polygon
from shapely.ops import unary_union

from src.pdf_parser.pdf_utils.base import TextBlock


class BaseLayoutExtractor(ABC):
    @abstractmethod
    def get_layout(self, model):
        pass


class LayoutParserExtractor(BaseLayoutExtractor):
    """Get layout from the image of a page using layoutparser Document-AI model.

    Attributes:
        image: The image of the page.
    """

    def __init__(self, image, model=None):
        self.image = np.array(image)

    def get_layout(self, model) -> lp.Layout:
        """Get layout from the image of a page using layoutparser Document-AI model.

        Returns:
            A layoutparser Layout object of text blocks.
        """
        layout = model.detect(self.image)  # perform computer vision
        return layout


class LayoutDisambiguator(LayoutParserExtractor):
    """Heuristics to disambiguate the layout from layoutparser computer vision models.

    Intent is to handle the following (non-exhaustive) cases:
        - Disambiguate nested blocks (using box confidence scores and empirically driven box type hierarchies).

    Attributes:
        image: The image of the page.
        model: The layoutparser model to use.
        layout_blocks_unfiltered: The layoutparser layout object of text blocks.
        layout_blocks: The layoutparser layout object of text blocks with a confidence score above the
        restrictive_theshold: The minimum confidence score for a box to be considered part of the layour in a strict model.
    """

    def __init__(
        self,
        image,
        model: lp.Detectron2LayoutModel,
        restrictive_threshold: float = 0.4,
    ):
        super().__init__(image)
        self.model = model
        self.layout_blocks_unfiltered = lp.Layout([b for b in model.detect(image)])
        self.restrictive_threshold = restrictive_threshold
        self.layout_blocks = [
            b for b in model.detect(image) if b.score >= restrictive_threshold
        ]

    @staticmethod
    def _lp_to_shapely_coords(
        coords: Tuple[float, float, float, float]
    ) -> List[Tuple[float, float]]:
        """Convert layoutparser coordinates to shapely format so that we can use convenient shapely ops.

        The coord format is as follows:

        [(x_top_left, y_top_left, x_bottom_right, y_bottom_right)] - > [(x_bottom_left, y_bottom_left), (x_top_left,
        y_top_left), (x_top_right, y_top_right), (x_bottom_right, y_bottom_right)]

        Args:
            coords: The layoutparser coordinates for the box.

        Returns:
            The shapely coordinates.
        """
        shapely_coords = [
            (coords[0], coords[1]),
            (coords[0], coords[3]),
            (coords[2], coords[3]),
            (coords[2], coords[1]),
        ]

        return shapely_coords

    @staticmethod
    def _shapely_to_lp_coords(
        coords: Tuple[
            Tuple[float, float],
            Tuple[float, float],
            Tuple[float, float],
            Tuple[float, float],
        ]
    ) -> Tuple[float, float, float, float]:
        """Convert shapely coordinates to layoutparser format so that we can use convenient layoutparser ops.

        ((x_bottom_left, y_bottom_left), (x_top_left, y_top_left), (x_top_right, y_top_right), (x_bottom_right,
        y_bottom_right)] - > [(x_top_left, y_top_left), (x_bottom_right, y_bottom_right))

        Args:
            coords: The shapely coordinates.

        Returns:
            The layoutparser coordinates.
        """
        lp_coords = (coords[0][0], coords[0][1], coords[2][0], coords[2][1])
        return lp_coords

    def _get_shapely_poly(self, box):
        return Polygon(self._lp_to_shapely_coords(box.block_1.coordinates))

    @property
    def layout(self):
        text_layout = lp.Layout(
            [
                box
                for box in self.layout_blocks
                if box.type in ["Text", "List", "Title", "Ambiguous"]
            ]
        )
        return text_layout

    @layout.setter
    def layout(self, blocks):
        """Implement setter so property is alterable."""
        self.layout_blocks = blocks

    @property
    def layout_unfiltered(self):
        text_layout = lp.Layout(
            [
                box
                for box in self.layout_blocks_unfiltered
                if box.type in ["Text", "List", "Title", "Ambiguous"]
            ]
        )
        return text_layout

    @layout_unfiltered.setter
    def layout_unfiltered(self, blocks):
        """Implement setter so property is alterable."""
        self.layout_blocks_unfiltered = blocks

    @property
    def unexplained_fractions(self) -> List[float]:
        """Return the fractions of the areas of blocks from an unfiltered perspective not covered by blocks from a strict perspective.

        This is useful because we want to find boxes that are not already accounted for by the strict model but that may contain
        useful text. Boxes detected in this way can be passed on to heuristics (e.g. linguistic features) to determine their category.


        Returns:
            A list of floats of the percentage unexplained coverage for each block in the unfiltered model.
        """

        unexplained_fractions = []
        permissive_polygons = [
            self._get_shapely_poly(box) for box in self.layout_unfiltered
        ]
        restrictive_polygons = [self._get_shapely_poly(box) for box in self.layout]
        for poly in permissive_polygons:
            poly_unexplained = poly.difference(unary_union(restrictive_polygons))
            area_unexplained = poly_unexplained.area
            area_total = poly.area
            frac_unexplained = area_unexplained / area_total
            unexplained_fractions.append(frac_unexplained)
        return unexplained_fractions

    def _combine_layouts(self, threshold: float) -> lp.Layout:
        """Add unexplained text boxes to the strict layouts to get a combined layout.

        Args:
            threshold: The unexplained area fraction above which to include boxes from the permissive perspective.

        Returns:
            The layout with boxes from the unfiltered perspective added if their areas aren't already sufficiently accounted for..
        """
        unexplained_fractions = self.unexplained_fractions
        strict_threshold_blocks = [b for b in self.layout]
        boxes_to_add = []
        for ix, box in enumerate(self.layout_unfiltered):
            # If the box's area is not "explained away" by the strict layout, add it to the combined layout with an
            # ambiguous type tag. We can use heuristics to determine its type downstream.
            if unexplained_fractions[ix] > threshold:
                box.block_1.type = "Ambiguous"
                boxes_to_add.append(box)
        combined_layout_list = strict_threshold_blocks + boxes_to_add
        new_layout = lp.Layout(combined_layout_list)
        self.layout = combined_layout_list
        return new_layout

    def _reduce_overlapping_boxes(
        self,
        box_1: lp.TextBlock,
        box_2: lp.TextBlock,
        direction: str = "vertical",
    ) -> Tuple[lp.TextBlock, lp.TextBlock]:
        """Reduce the size of overlapping boxes to elimate overlaps.

        If two boxes overlap in a given direction (vertical or horizontal), reduce the size of both in that
         direction by the minimal amount necessary to elimate overlaps.

        Args:
            box_1: The first box to compare. This box should be the upper/left box.
            box_2: The second box to compare. This box should be the lower/right box.
            direction: The direction to reduce the boxes in.

        Returns:
            The boxes with overlaps elimated.
        """
        if direction == "vertical":
            assert (
                box_1.block_1.coordinates[1] < box_2.block_1.coordinates[1]
            ), "box_1 should be the upper box."
            intersection_height = box_1.intersect(box_2).height
            rect_1 = lp.Rectangle(
                x_1=box_1.coordinates[0],
                y_1=box_1.coordinates[1],
                x_2=box_1.coordinates[2],
                y_2=box_1.coordinates[3] - intersection_height,
            )
            rect_2 = lp.Rectangle(
                x_1=box_2.coordinates[0],
                y_1=box_2.coordinates[1] + intersection_height,
                x_2=box_2.coordinates[2],
                y_2=box_2.coordinates[3],
            )
        elif direction == "horizontal":
            assert (
                box_1.block_1.coordinates[0] < box_2.block_1.coordinates[0]
            ), "box_1 should be the left box."
            intersection_width = box_1.intersect(box_2).width
            rect_1 = lp.Rectangle(
                x_1=box_1.coordinates[0],
                y_1=box_1.coordinates[1],
                x_2=box_1.coordinates[2] - intersection_width,
                y_2=box_1.coordinates[3],
            )
            rect_2 = lp.Rectangle(
                x_1=box_2.coordinates[0] + intersection_width,
                y_1=box_2.coordinates[1],
                x_2=box_2.coordinates[2],
                y_2=box_2.coordinates[3],
            )
        return rect_1, rect_2

    def _reduce_all_overlapping_boxes(
        self, blocks: lp.Layout, reduction_direction: str = "vertical"
    ) -> lp.Layout:
        """Eliminate all overlapping boxes by reducing their size by the minimal amount necessary.

        In general, for every pair of rectangular boxes with coordinates of
        the form (x_top_left, y_top_left, x_bottom_right, y_bottom_right),
        we want to reshape them to elimate the intersecting regions in the
        alighnment direction. For example, if we want to elimate overlaps of
        the following two rectangles with a prior that vertical overlaps should
        be removed, the transformation should be

        [(0,0,3,3),(1,1,2,4)] -> [(0,0,3,1), (1,3,2,4)]

        Args:
            blocks: The blocks to reduce.
            reduction_direction: The direction to reduce the boxes in.

        Returns:
            The new layout with blocks having no overlapping coordinates.
        """

        for box_1, box_2 in zip(blocks, blocks):
            if box_1 == box_2:
                continue
            else:
                if box_1.intersect(box_2).area > 0:
                    if reduction_direction == "vertical":
                        # check which box is upper and which is lower
                        if box_1.coordinates[3] < box_2.coordinates[3]:
                            rect_1, rect_2 = self._reduce_overlapping_boxes(
                                box_1, box_2, direction=reduction_direction
                            )
                        else:
                            rect_1, rect_2 = self._reduce_overlapping_boxes(
                                box_2, box_1, direction=reduction_direction
                            )
                    elif reduction_direction == "horizontal":
                        # check which box is left and which is right
                        if box_1.coordinates[2] < box_2.coordinates[2]:
                            rect_1, rect_2 = self._reduce_overlapping_boxes(
                                box_1, box_2, direction=reduction_direction
                            )
                        else:
                            rect_1, rect_2 = self._reduce_overlapping_boxes(
                                box_2, box_1, direction=reduction_direction
                            )
                    else:
                        raise ValueError(
                            "reduction_direction must be either 'vertical' or 'horizontal'"
                        )
                    box_1.block_1 = rect_1
                    box_2.block_1 = rect_2
        return blocks

    def _unnest_boxes(self, unnest_inflation_factor: float = 0.2) -> lp.Layout:
        """
        Recursively Unnest boxes.

        Args:
            unnest_inflation_factor: The amount to inflate the unnested box by (i.e. a soft margin coefficient).

        Returns:
            The unnested boxes.
        """
        # The loop checks each block for containment within other blocks.
        # Contained blocks are removed if they have lower confidence scores than their parents;
        # otherwise, the parent is removed. The process continues until there are no contained blocks.
        # There are potentially nestings within nestings, hence the rather complicated loop.
        # A recursion might be more elegant, leaving it as a TODO.
        stop_cond = True
        counter = 0  # count num contained blocks in every run through of all pair combinations to calculate stop
        # condition.
        disambiguated_layout = self.layout
        while stop_cond:
            for ix, box_1 in enumerate(disambiguated_layout):
                for ix2, box_2 in enumerate(disambiguated_layout):
                    if box_1 == box_2:
                        continue
                    else:
                        # Ass a soft-margin for the is_in function to allow for some leeway in the containment check.
                        height_inflation = unnest_inflation_factor * box_2.height
                        width_inflation = unnest_inflation_factor * box_2.width
                        soft_margin = {
                            "top": height_inflation,
                            "bottom": height_inflation,
                            "left": width_inflation,
                            "right": width_inflation,
                        }
                        if box_1.is_in(box_2, soft_margin):
                            counter += 1
                            # Remove the box the model is less confident about.
                            if box_1.score > box_2.score:
                                remove_ix = ix2
                            else:
                                remove_ix = ix
                            disambiguated_layout = lp.Layout(
                                [
                                    box
                                    for index, box in enumerate(disambiguated_layout)
                                    if index != remove_ix
                                ]
                            )
                # stop condition: no contained blocks
                if counter == 0:
                    stop_cond = False
                counter = 0
        return disambiguated_layout

    # TODO: Instead of this, we can use Google OCR to extract all text from a page and index everything not indexed.
    def _create_unknown_text_blocks(
        self, threshold: float = 0.25, vertical: bool = True
    ) -> lp.Layout:
        """Create text blocks by grabbing blocks with low confidence scores and subtracting intersections
         with the current layout to create new text blocks.

        This has utility beyond the _combine_layouts method because it extracts text even in cases where
        there are huge text blocks with high emounts of explained area so they aren't caught by the
        _combine_layouts method.

        Args:
            threshold: The confidence threshold to use for adding unknown text blocks.
            vertical: Whether to add unknown text blocks based on vertical or horizontal reading order assumptions
             (nuance this later, see TODO).

        Returns:
            The layout with unidentified (but probable) text blocks added.
        """
        # TODO: Ideally this should be done after we've detected the reading order
        #  so we can guess that we should create 2 boxes in a missing area instead
        #  of one, if there are two columns, for example. This is a manifestation of
        #  the more general that order of processing matters. The best way of solving
        #  this is probably to use class composition. Leaving this for now as this should
        #  handle the majority of cases.
        all_text_blocks = self.layout_unfiltered
        text_blocks_below_threshold = lp.Layout(
            [b for b in all_text_blocks if b.score < threshold]
        )
        # Loop through every possible combination of two text blocks with one from an unfiltered perspective and another
        # from all perspectives. Subtract the vertical coordinates of overlap from the unfiltered perspective to get coordinates
        # for a new text block.
        new_text_blocks = []
        if len(text_blocks_below_threshold) > 0:
            for block_1 in text_blocks_below_threshold:
                unaccounted_line = LineString(
                    [(block_1.coordinates[1], 0), (block_1.coordinates[3], 0)]
                )
                for block_2 in all_text_blocks:
                    if block_1.coordinates == block_2.coordinates:
                        continue
                    else:
                        line_2 = LineString(
                            [
                                (block_2.coordinates[1], 0),
                                (block_2.coordinates[3], 0),
                            ]
                        )
                        line_intersection = line_2.intersection(unaccounted_line)
                        if line_intersection.is_empty:
                            continue
                        else:
                            unaccounted_line = line_2.difference(line_intersection)
                            if unaccounted_line.bounds == line_2.bounds:
                                continue
                y1, y2 = unaccounted_line.bounds[0], unaccounted_line.bounds[2]
                x1, x2 = block_1.coordinates[0], block_1.coordinates[2]
                unaccounted_block_shape = lp.Rectangle(x1, y1, x2, y2)
                # TODO: include metadata such as a score in the text block?
                unaccounted_text_block = lp.TextBlock(
                    unaccounted_block_shape, type="Ambiguous"
                )
                new_text_blocks.append(unaccounted_text_block)
        else:
            pass

        self.layout = [*[b for b in self.layout], *new_text_blocks]
        return self.layout

    def _calculate_coverage(self):
        """Calculate the percentage of the page that is covered by text blocks."""
        image_array = np.array(self.image)
        coverage = (
            100
            * sum([box.area for box in self.layout])
            / (image_array.shape[0] * image_array.shape[1])
        )
        return coverage

    def disambiguate_layout(
        self, unnest_inflation_factor: float = 0.2, threshold: float = 0.35
    ) -> lp.Layout:
        """Disambiguate the layout by unnesting nested boxes using heuristics and removing overlaps for OCR.

        Where boxes are nested within other boxes, the outermost box is taken unless the model
        is more confident in the inner box, in which case the inner box is taken. This is
        repeated until there are no more boxes within other boxes.

        Args:
            unnest_inflation_factor: The amount by which to inflate the bounding boxes when checking for containment.
            threshold: The confidence threshold to use for adding unknown text blocks.

        Returns:
            The disambiguated layout.
        """
        # Unnest boxes so that there are no boxes within other boxes.
        disambiguated_layout = self._unnest_boxes(
            unnest_inflation_factor=unnest_inflation_factor
        )
        # TODO: These functions are buggy/not working as anticipated.
        #  Fix them and then uncomment.
        # disambiguated_layout = self._create_unknown_text_blocks(
        #     threshold=threshold, vertical=True
        # )
        # # Ensure the remaining rectangles have no overlap for OCR.
        # disambiguated_layout = self._reduce_all_overlapping_boxes(
        #     disambiguated_layout, reduction_direction="vertical"
        # )
        # disambiguated_layout = self._reduce_all_overlapping_boxes(
        #     disambiguated_layout, reduction_direction="horizontal"
        # )
        return disambiguated_layout

    def disambiguate_layout_advanced(self) -> lp.Layout:
        """Disambiguate the blocks using a hierarchy of heuristics.

        There are a number of hierarchical heuristics we can apply to disambiguate blocks that are fully
        nested within other blocks:
            - If a list block is contained within another list blocks, the outer list supercedes the inner list.
            - If a list block is within a text block, take the text block on the grounds that it will include
               more text. From a search perspective, this is more desirable than missing out on a list, since
               it isn't clear how much information we can parse from the fact that something is a list anyway.
            - If a text block is contained within another text block, the outer text block supercedes the inner text block.
            - If a text block is contained within a list block, the list supersedes the text.
            - After these steps, if

        For now, all of these cases can be handled by simply removing the nested block. But we may need
        more sophisticated heuristics in the future (an empirical question).

        Returns:
            The disambiguated layout.
        """
        raise NotImplementedError(
            "This method is not yet implemented. Placeholder for more advanced heuristics later, see docstring."
        )


class DetectReadingOrder:
    """Helper class for detecting the layout of content from a layoutparser computer vision models
    using visual heuristics.

    Intent is to handle the following (non-exhaustive) cases:
        - Reading order inference.

    Attributes:
        layout: The layoutparser layout of the page.
        non_text_blocks: The non-text elements from the image.
    """

    def __init__(
        self,
        layout: lp.Layout,
    ):
        self.layout = layout
        self.reordered_ocr_blocks = None

    @property
    def ocr_blocks(self) -> lp.Layout:
        return lp.Layout(
            [b for b in self.layout if b.type in ["Text", "List", "Title", "Ambiguous"]]
        )

    @property
    def non_text_blocks(self) -> lp.Layout:
        return lp.Layout(
            [
                b
                for b in self.layout
                if b.type not in ["Text", "List", "Title", "Ambiguous"]
            ]
        )

    @staticmethod
    def _calc_frac_overlap(block_1: lp.TextBlock, block_2: lp.TextBlock) -> float:
        """Calculate the fraction of overlap between two blocks.

        This is useful for splitting a layout into columns to infer the reading order.

        Args:
            block_1: The first text block.
            block_2: The second text block.

        Returns:
            The percentage of overlap between the two blocks.
        """
        union_width = block_1.union(block_2).width
        intersection_width = block_1.intersect(block_2).width
        return intersection_width / union_width

    def _infer_column_groups(self, threshold: float = 0.95):
        """Group text blocks into columns depending on an x-overlap threshold.

        Assumption is that blocks with a given x-overlap are in the same column. This
        is a heuristic encoding of a reading order prior.

        Args:
            threshold: The threshold for the percentage of overlap in the x-direction.

        Returns:
            An array of text block groups.
        """
        dd = defaultdict(
            list
        )  # keys are the text block index; values are the other indices that are inferred to be in the same reading column.
        # Calculate the percentage overlap in the x-direction of every text block with every other text block.
        for ix, i in enumerate(self.ocr_blocks):
            for j in self.ocr_blocks:
                dd[ix].append(self._calc_frac_overlap(i, j))
        df_overlap = pd.DataFrame(dd)
        df_overlap = (
            df_overlap > threshold
        )  # same x-column if intersection over union > threshold
        # For each block, get a list of shared blocks.
        shared_blocks = df_overlap.apply(
            lambda row: str(row[row].index.tolist()), axis=1
        )
        # put into numeric groups for cleanness.
        column_groups = pd.factorize(shared_blocks)[0]
        return column_groups

    def infer_reading_order(self, threshold: float = 0.95) -> lp.Layout:
        """Infer the reading order of the text blocks.

        Encodes the following prior: the intended reading order is to read
        all rows of a column first, then move to the next column.

        Args:
            threshold: The threshold for the percentage of overlap in the x-direction to infer
                that two blocks are in the same column.

        Returns:
            The text blocks with the inferred reading order.
        """
        ocr_blocks = self.ocr_blocks
        column_groups = self._infer_column_groups(threshold)
        df_text_blocks = ocr_blocks.to_dataframe()
        df_text_blocks["group"] = column_groups
        df_text_blocks["x_1_min"] = df_text_blocks.groupby("group")["x_1"].transform(
            min
        )

        # split df into groups, sort values by y_1, then concatenate groups according to x_1.
        df_natural_reading_order = df_text_blocks.sort_values(
            ["x_1_min", "y_1"], ascending=[True, True]
        )
        reading_order = df_natural_reading_order.index.tolist()
        reordered_ocr_blocks = lp.Layout([ocr_blocks[i] for i in reading_order])
        return reordered_ocr_blocks


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
        layout: lp.Layout,
        ocr_agent: Union[ocr.TesseractAgent, ocr.GCVAgent],
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
        block: lp.TextBlock,
        left_pad: int = 15,
        right_pad: int = 5,
        top_pad: int = 5,
        bottom_pad: int = 5,
    ) -> Tuple[lp.TextBlock, Optional[str]]:
        """
        Perform OCR on a block of text.

        Args:
            image: The image to perform OCR on.
            block: The block to set the text of.
            left_pad: The number of pixels to pad the left side of the block.
            right_pad: The number of pixels to pad the right side of the block.
            top_pad: The number of pixels to pad the top of the block.
            bottom_pad: The number of pixels to pad the bottom of the block.

        Returns:
            lp.TextBlock: text block with the text set.
            str: the language of the text or None if the OCR processor doesn't support language detection.
        """
        # TODO: THis won't work currently because the image isn't part of the class.
        # Pad to improve OCR accuracy as it's fairly tight.

        segment_image = block.pad(
            left=left_pad, right=right_pad, top=top_pad, bottom=bottom_pad
        ).crop_image(image)

        # Perform OCR
        if isinstance(self.ocr_agent, ocr.TesseractAgent):
            language = None
            text = self.ocr_agent.detect(segment_image, return_only_text=True)

        elif isinstance(self.ocr_agent, ocr.GCVAgent):
            gcv_response = self.ocr_agent.detect(segment_image, return_response=True)
            text = gcv_response.full_text_annotation.text

            # We assume one language per text block here which seems reasonable, but may not always be true.
            language = (
                gcv_response.full_text_annotation.pages[0]
                .property.detected_languages[0]
                .language_code
            )

        # Save OCR result
        block_with_text = block.set(text=text)

        return block_with_text, language

    def process_layout(self) -> List[TextBlock]:
        """
        Get text for the text blocks in the layout, and return a `Page` object with text retrieved, and language and text block IDs set per text block.

        :return: list of text blocks
        """
        text_blocks = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for block_idx, block in enumerate(self.layout):
                future = executor.submit(self._perform_ocr, self.image, block)
                block_with_text, block_language = future.result()
                if block.type == "Ambiguous":
                    block_with_text.type = self._infer_block_type(block)

                text_block_id = f"p_{self.page_number}_b_{block_idx}"
                text_block = TextBlock.from_layoutparser(block_with_text, text_block_id)
                text_block.language = block_language

                text_blocks.append(text_block)

        return text_blocks
