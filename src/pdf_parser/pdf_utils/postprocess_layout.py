import itertools
from collections import defaultdict
from typing import List
from layoutparser import Layout, TextBlock, Rectangle
import pandas as pd


def ocr_blocks(layout: Layout) -> Layout:
    """Return the text blocks for OCR from the layout.

    Args:
        layout: The layout to extract the text blocks from.

    Returns:
        The text blocks from the layout.
    """
    return Layout(
        [
            b
            for b in layout
            if b.type in ["Text", "List", "Title", "Ambiguous", "Inferred from gaps"]
        ]
    )


def group_blocks_into_columns(
    blocks: Layout, column_overlap_threshold: float = 0.25
) -> pd.DataFrame:
    """Group the blocks into columns.

    This is a prerequisite for encoding the following prior: the intended reading order is to read
    all rows of a column first, then move to the next column.

    Args:
        blocks: The text blocks to group into columns.
        column_overlap_threshold: The threshold for the percentage of overlap in the x-direction to infer
            that two blocks are in the same column.

    Returns:
        A dataframe with the text blocks grouped into columns.
    """
    column_groups = infer_column_groups(blocks, column_overlap_threshold)
    df_text_blocks = blocks.to_dataframe()
    df_text_blocks["group"] = column_groups
    df_text_blocks["x_1_min"] = df_text_blocks.groupby("group")["x_1"].transform(min)
    return df_text_blocks


def split_layout_into_cols(blocks: Layout) -> List[Layout]:
    """Group the OCR blocks into columns.

    Args:
        blocks: The text blocks to group into columns.

    Returns:
        A list with each element a column of text blocks.
    """
    # group blocks into columns and return the layout of each column in a list.
    column_layouts_df = group_blocks_into_columns(blocks)
    column_layouts = []
    for column, df in column_layouts_df.groupby("group"):
        keep_index = df.index
        original_blocks = [blocks[i] for i in keep_index]
        column_layouts.append(Layout(original_blocks))
    return column_layouts


def calc_frac_overlap(block_1: TextBlock, block_2: TextBlock) -> float:
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


def infer_missing_blocks_from_gaps(
    column_blocks: Layout,
    page_height: int,
    height_threshold: int = 50,
    start_pixel_buffer: int = 10,
    end_pixel_buffer: int = 10,
) -> List[TextBlock]:
    """Infer the missing blocks in columns by checking for sufficiently large gaps.

    Args:
        column_blocks: The text blocks to infer the missing blocks from.
        page_height: The height of the page in num pixels.
        height_threshold: The number of pixels for a missing blocks to be inferred (avoiding normal whitespace).
        start_pixel_buffer: num pixels to ignore at the start of the column (avoiding non-columnar titles etc).
        end_pixel_buffer: The number of pixels to ignore at the end of the column (heuristic to exclude footers, etc).

    Returns:
        The text blocks with the inferred missing blocks.
    """
    # Make sure the blocks are sorted by y_1.
    column_blocks = column_blocks.sort(key=lambda b: b.coordinates[1])
    x1 = min([b.coordinates[0] for b in column_blocks])
    x2 = max([b.coordinates[2] for b in column_blocks])
    # Iteratively fill in gaps between subsequent blocks.
    new_blocks = []

    for ix in range(len(column_blocks)):
        if ix < len(column_blocks) - 1:
            y1_new = column_blocks[ix].coordinates[3]
            y2_new = column_blocks[ix + 1].coordinates[1]
            height_new = y2_new - y1_new
            if height_new >= height_threshold:
                new_block_shape = Rectangle(x1, y1_new, x2, y2_new)
                new_block = TextBlock(
                    new_block_shape, type="Inferred from gaps", score=1.0
                )
                new_blocks.append(new_block)
        # If there is a gap at the start of the column starting from the pixel buffer, add a block.
        if (ix == 0) & (
            column_blocks[ix].coordinates[1] - start_pixel_buffer > height_threshold
        ):
            new_blocks.append(
                TextBlock(
                    Rectangle(
                        x1,
                        start_pixel_buffer,
                        x2,
                        column_blocks[ix].coordinates[1],
                    ),
                    type="Inferred from gaps",
                    score=1.0,
                )
            )
        # If there is a gap at the end of the column ending at the pixel buffer, add a block. For this, we need
        # the page length.
        elif (ix == len(column_blocks) - 1) & (
            page_height - column_blocks[ix].coordinates[3] - end_pixel_buffer
            > height_threshold
        ):
            new_blocks.append(
                TextBlock(
                    Rectangle(
                        x1,
                        column_blocks[ix].coordinates[3],
                        x2,
                        page_height - end_pixel_buffer,
                    ),
                    type="Inferred from gaps",
                    score=1.0,
                )
            )
    return new_blocks


def infer_column_groups(blocks: Layout, column_overlap_threshold: float = 0.20):
    """Group text blocks into columns depending on an x-overlap threshold.

    Assumption is that blocks with a given x-overlap are in the same column. This
    is a heuristic encoding of a reading order prior.

    Args:
        blocks: The text blocks to group into columns.
        column_overlap_threshold: The threshold for the percentage of overlap in the x-direction.

    Returns:
        An array of text block groups.
    """
    dd = defaultdict(
        list
    )  # keys are the text block index; values are the other indices that are inferred to be in the same reading column.
    # Calculate the percentage overlap in the x-direction of every text block with every other text block.
    for ix, i in enumerate(blocks):
        for j in blocks:
            dd[ix].append(calc_frac_overlap(i, j))
    df_overlap = pd.DataFrame(dd)
    df_overlap = (
        df_overlap > column_overlap_threshold
    )  # same x-column if intersection over union > threshold
    # For each block, get a list of blocks indices in the same column.
    shared_col_idxs = df_overlap.apply(lambda row: str(row[row].index.tolist()), axis=1)
    # Get a list with each element a list of the block indices for each unique group.
    unique_shared_col_idxs = [eval(i) for i in shared_col_idxs.unique()]

    # In some cases, the heuristic above creates column groups with overlapping block elements.
    # This happens when the overlap threshold for inclusion in the same column is exceeded for some but not all
    # blocks. In this case, we should still infer that the blocks are in the same column. To do this, we
    # iteratively merge groups if they have overlapping elements, then remove duplicates within the groups
    # first, then remove duplicates across groups.
    col_groups = []
    for ix, lst in enumerate(unique_shared_col_idxs):
        col_groups.append(lst)
        for lst_2 in unique_shared_col_idxs:
            if lst == lst_2:
                continue
            # if there are shared elements, merge the two groups.
            if set(lst).intersection(lst_2):
                col_groups[ix].extend(lst_2)

    # Remove duplicate block indices in each group.
    deduplicated_groups = [list(set(ls)) for ls in col_groups]
    # Remove duplicate groups.
    deduplicated_groups.sort()
    final_column_groups = list(k for k, _ in itertools.groupby(deduplicated_groups))

    # Now return a group index for each block. (e.g if block 1 is in group 0, block 2 is in group 1, etc.)
    block_group_idxs = []
    for num in range(len(blocks)):
        for ix, group_list in enumerate(final_column_groups):
            if num in group_list:
                block_group_idxs.append(ix)

    return block_group_idxs


def filter_inferred_blocks(blocks: Layout, remove_threshold: float = 0.20) -> Layout:
    """Remove inferred blocks if they are covered by other blocks. Heuristic.

    Args:
        blocks: The text blocks to filter.
        remove_threshold: The threshold of area overlap above which to remove the inferred block.

    Returns:
        Layout with Inferred blocks removed if more than remove_threshold of their area
        is accounted for by other blocks.

    """
    # For blocks with type "Inferred from gaps", remove them if more than the removal threshold
    # is accounted for by other blocks.
    ixs_to_remove = []
    for ix, block in enumerate(blocks):
        if block.type != "Inferred from gaps":
            continue
        else:
            block_area = block.area
            accounted_for_area = 0
            for other_block in blocks:
                if block == other_block:
                    continue
                # Assumption that the other blocks do not overlap. This is almost always true.
                intersect_area = block.intersect(other_block).area
                if intersect_area > 0:
                    accounted_for_area += intersect_area
            accounted_for_fraction = accounted_for_area / block_area
            if accounted_for_fraction > remove_threshold:
                ixs_to_remove.append(ix)

    return Layout([b for ix, b in enumerate(blocks) if ix not in ixs_to_remove])


# def postprocess(
#     inference_method: str = "gaps",
# ) -> Layout:
#     """Infer probable text blocks that haven't been detected and infer reading order.
#
#     This has utility beyond the _combine_layouts method because it extracts text even in cases where
#     there are huge text blocks with high emounts of explained area so they aren't caught by the
#     _combine_layouts method.
#
#     Note the default inference method is "gaps". This method fills in gaps in previously inferred columns
#     if the gaps are too big to be white space. The risk is false positives (i.e. genuine whitespace that is
#     larger than the threshold) or the capture of figures/tables that the parser missed. This is rare, but can
#     be corrected for downstream using heuristics from the text returned by OCR if need be. The alternative method,
#     perspectives, tried to fill gaps in areas covered by detectron2 text blocks with low confidence scores. Empirically,
#     this method was less effective than the gaps method because detectron2 sometimes fails to detect text blocks at all,
#     even with low confidence.
#
#     Args:
#         inference_method: The method to use for inferring the missing blocks. Options are "gaps" and "threshold". See docstrings for details.
#
#     Returns:
#         The layout with unidentified (but probable) text blocks added.
#     """
#     text_blocks = ocr_blocks
#     if len(text_blocks) == 0:
#         return Layout([])
#     # group blocks into columns and return the layout of each column in a list.
#     column_layouts = split_layout_into_cols(text_blocks)
#
#     if inference_method == "gaps":  # infer blocks based on gaps in each column.
#         new_text_blocks = []
#         for layout in column_layouts:
#             new_text_blocks.append(infer_missing_blocks_from_gaps(layout))
#         # flatten the list of lists
#         new_text_blocks = [item for sublist in new_text_blocks for item in sublist]
#     elif (
#         inference_method == "perspective"
#     ):  # infer blocks based on coverage of gaps by boxes from a more permissive perspective.
#         assert layout_unfiltered is not None
#         text_blocks_permissive = Layout(
#             [b for b in layout_unfiltered if b.score >= threshold]
#         )
#
#         # Assign the unfiltered blocks to different layouts depending on which column they're in (if any).
#         unfiltered_column_layouts = assign_new_blocks_to_columns(
#             text_blocks, text_blocks_permissive
#         )
#         # Get the layout of all columns with all perspectives.
#         additional_layouts_all = [
#             column_layouts[i] + unfiltered_column_layouts[i]
#             for i in range(len(column_layouts))
#         ]
#
#         new_text_blocks = infer_missing_blocks_from_perspectives(
#             column_layouts, additional_layouts_all
#         )
#     else:
#         raise ValueError("Inference method must be either 'gaps' or 'perspective'.")
#
#     # reorder the new inferred text blocks and create a final layout object.
#     unordered_layout = Layout([*[b for b in layout], *new_text_blocks])
#     if len(unordered_layout) == 0:
#         return Layout([])
#     df_text_blocks = group_blocks_into_columns(unordered_layout)
#     df_natural_reading_order = df_text_blocks.sort_values(
#         ["x_1_min", "y_1"], ascending=[True, True]
#     )
#     reading_order = df_natural_reading_order.index.tolist()
#     layout = Layout([unordered_layout[i] for i in reading_order])
#     layout = filter_inferred_blocks(layout)
#     return layout
