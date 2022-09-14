from pathlib import Path
import hashlib

import click
import numpy as np
import layoutparser as lp
import loguru
from cloudpathlib import CloudPath
from tqdm import tqdm

from src.pdf_parser.pdf_utils.parsing_utils import (
    OCRProcessor,
    LayoutDisambiguator,
    DetectReadingOrder,
)

from src.pdf_parser.pdf_utils.base import Document, Page


@click.command()
@click.option(
    "-i",
    "--input-dir",
    type=click.Path(exists=True),
    required=True,
    help="The directory to read PDFs from.",
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--ocr-agent", type=click.Choice(["tesseract", "gcv"]), required=True, default="gcv"
)
@click.option(
    "--device", type=click.Choice(["cuda", "cpu"]), required=True, default="cpu"
)
@click.option("-l", "--test-limit", type=int, default=1)
@click.option(
    "-m",
    "--model",
    type=str,
    required=True,
    help="The model to use for OCR.",
    default="mask_rcnn_X_101_32x8d_FPN_3x",  # powerful detectron-2 model.
)
@click.option(
    "-ts", "--model-threshold-restrictive", type=float, default=0.5
)  # Hyperparam, set up config
def run_cli(
    input_dir: Path,
    output_dir: Path,
    test_limit: int,
    ocr_agent: str,
    model: str,
    model_threshold_restrictive: float = 0.4,
    device: str = "cpu",
) -> None:
    """
    Run cli to extract semi-structured JSON from document-AI + OCR.

    Args:
        input_dir: The directory containing the PDFs to parse.
        output_dir: The directory to write the parsed PDFs to.
        ocr_agent: The OCR agent to use.
        test_limit: Place a limit on the number of PDFs to parse - useful for testing.
        model: The document AI model to use.
        model_threshold_restrictive: The threshold to use for the document AI model.
        device: The device to use for the document AI model.
    """
    loguru.logger.info(f"Using {ocr_agent} OCR agent.")
    if ocr_agent == "gcv":
        loguru.logger.warning(
            "THIS IS COSTING MONEY/CREDITS - BE CAREFUL WHEN TESTING."
        )
    loguru.logger.info(f"Using {model} model.")

    loguru.logger.info(f"Reading from {input_dir}.")
    if input_dir.startswith("s3://"):
        input_dir = CloudPath(input_dir)
    else:
        input_dir = Path(input_dir)

    if output_dir.startswith("s3://"):
        output_dir = CloudPath(output_dir)
    else:
        output_dir = Path(output_dir)

    # TODO: We may want to make this an option, but for now just use Detectron by default as we are unlikely
    #  to change this unless we start labelling by ourselves.
    def _get_detectron_model() -> lp.Detectron2LayoutModel:
        return lp.Detectron2LayoutModel(
            config_path=f"lp://PubLayNet/{model}",  # In model catalog,
            label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
            device=device,
        )

    model = _get_detectron_model()

    if ocr_agent == "tesseract":
        ocr_agent = lp.TesseractAgent()
    elif ocr_agent == "gcv":
        ocr_agent = lp.GCVAgent()

    loguru.logger.info("Iterating through files and parsing pdf content from pages.")
    # test_pdfs_and_pages=[('SMR-2012-01-01-Environmental code_e4596ec8f85c2cd4576593579c0daf8d.pdf',
    #   83),
    #  ('ERI-2009-12-25-Five Year Indicative Development Plan (FYIDP)_4201842f0df84d68793f8dc1c5d21456.pdf',
    #   14),
    #  (
    #  'CZE-2016-01-01-State Environmental Policy of the Czech Republic 2012–2020, as amended by the 2016 update_33bb9056d108bce3d856c980fb5b6648.pdf',
    #  26),
    #  ('DEU-2020-10-06-National Hydrogen Strategy (NWS)_c606f1e99ddbc316b8cb80e22645d020.pdf',
    #   22),
    #  ('ERI-2009-12-25-Five Year Indicative Development Plan (FYIDP)_4201842f0df84d68793f8dc1c5d21456.pdf',
    #   27),
    #  (
    #  'CZE-2016-01-01-State Environmental Policy of the Czech Republic 2012–2020, as amended by the 2016 update_33bb9056d108bce3d856c980fb5b6648.pdf',
    #  89),
    #  ('SMR-2012-01-01-Environmental code_e4596ec8f85c2cd4576593579c0daf8d.pdf',
    #   174),
    #  (
    #  'CZE-2016-01-01-State Environmental Policy of the Czech Republic 2012–2020, as amended by the 2016 update_33bb9056d108bce3d856c980fb5b6648.pdf',
    #  66),
    #  ('SMR-2012-01-01-Environmental code_e4596ec8f85c2cd4576593579c0daf8d.pdf',
    #   87),
    #  ('DEU-2020-10-06-National Hydrogen Strategy (NWS)_c606f1e99ddbc316b8cb80e22645d020.pdf',
    #   19)]
    for page_idx, file in enumerate(tqdm(input_dir.glob("*.pdf"), desc="Files")):
        if page_idx >= test_limit:
            break
        # if file.name not in [pdf[0][0] for pdf in test_pdfs_and_pages]:
        #     continue
        page_layouts, pdf_images = lp.load_pdf(file, load_images=True)
        pages = []
        for page_idx, image in tqdm(
            enumerate(pdf_images), total=len(pdf_images), desc=file.name
        ):
            # Maybe we should always pass a layout object into the PageParser class.
            layout_disambiguator = LayoutDisambiguator(
                image, model, model_threshold_restrictive
            )
            disambiguated_layout = layout_disambiguator.disambiguate_layout()
            if len(disambiguated_layout) == 0:
                loguru.logger.info(f"No layout found for page {page_idx}.")
                continue

            # disambiguated_layout = layout_disambiguator.recursive_disambiguator()
            reading_order_detector = DetectReadingOrder(disambiguated_layout)
            ocr_blocks = reading_order_detector.infer_reading_order()
            ocr_processor = OCRProcessor(
                image=np.array(image),
                page_number=page_idx,
                layout=ocr_blocks,
                ocr_agent=ocr_agent,
            )
            text_blocks = ocr_processor.process_layout()

            page_dimensions = (
                page_layouts[page_idx].page_data["width"],
                page_layouts[page_idx].page_data["height"],
            )
            page = Page(
                text_blocks=text_blocks,
                dimensions=page_dimensions,
                page_number=page_idx,
            )

            pages.append(page)

        document = Document(
            pages=pages,
            filename=file.stem,
            md5hash=hashlib.md5(file.read_bytes()).hexdigest(),
        ).set_languages(min_language_proportion=0.4)

        output_path = output_dir / f"{file.stem}.json"

        with open(output_path, "w") as f:
            f.write(document.json(indent=4, ensure_ascii=False))

        loguru.logger.info(f"Saved {output_path.name} to {output_dir}.")


if __name__ == "__main__":
    run_cli()
