import pytest
from layoutparser import load_pdf, Detectron2LayoutModel
from pathlib import Path


@pytest.fixture
def base_model():
    return Detectron2LayoutModel(
        config_path="lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x",  # See model catalog,
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
        device="cpu",
    )


@pytest.fixture
def test_image(base_model):
    """Load a page with useful test properties."""
    pdf_path = (
        Path(__file__).parent
        / "data"
        / "BRB-2019-12-25-National Energy Policy 2019-2030_19fbfbb2c35d8f43bfa1b8c3219605b4.pdf"
    )
    _, pdf_images = load_pdf(pdf_path, load_images=True)
    pdf_image = pdf_images[47]
    return pdf_image


@pytest.fixture
def test_layout(test_image, base_model):
    return base_model.detect(test_image)
