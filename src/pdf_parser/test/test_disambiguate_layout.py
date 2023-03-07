import pytest
from pathlib import Path
from layoutparser import load_pdf, Layout, TextBlock, Rectangle, Detectron2LayoutModel
from shapely.geometry import Polygon

from src.pdf_parser.pdf_utils.disambiguate_layout import (
    split_layout,
    remove_nested_boxes,
    calculate_unexplained_fractions,
    lp_coords_to_shapely_polygon,
    LayoutWithFractions,
)


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


@pytest.fixture
def layout_permissive():
    return Layout(
        [
            TextBlock(
                block=Rectangle(
                    x_1=311.2409362792969,
                    y_1=438.8702697753906,
                    x_2=566.844970703125,
                    y_2=619.61328125,
                ),
                text=None,
                id=None,
                type="Figure",
                parent=None,
                next=None,
                score=0.4856751263141632,
            ),
            TextBlock(
                block=Rectangle(
                    x_1=49.70960235595703,
                    y_1=105.66661834716797,
                    x_2=389.490234375,
                    y_2=214.9551544189453,
                ),
                text=None,
                id=None,
                type="Figure",
                parent=None,
                next=None,
                score=0.26342472434043884,
            ),
            TextBlock(
                block=Rectangle(
                    x_1=65.97888946533203,
                    y_1=96.48314666748047,
                    x_2=293.3446960449219,
                    y_2=565.2528076171875,
                ),
                text=None,
                id=None,
                type="List",
                parent=None,
                next=None,
                score=0.11599220335483551,
            ),
            TextBlock(
                block=Rectangle(
                    x_1=312.40087890625,
                    y_1=630.3326416015625,
                    x_2=529.0540161132812,
                    y_2=641.197509765625,
                ),
                text=None,
                id=None,
                type="Title",
                parent=None,
                next=None,
                score=0.0948110818862915,
            ),
            TextBlock(
                block=Rectangle(
                    x_1=454.49688720703125,
                    y_1=442.1412658691406,
                    x_2=547.2313842773438,
                    y_2=466.0764465332031,
                ),
                text=None,
                id=None,
                type="Text",
                parent=None,
                next=None,
                score=0.05058730021119118,
            ),
        ]
    )


@pytest.fixture
def layout_permissive_with_fractions(layout_permissive):
    layout_permissive_with_fractions = LayoutWithFractions(
        layout=layout_permissive,
        unexplained_fractions=[
            0.05624759027532935,
            0.22591804660118936,
            0.22530762135686025,
            0.06766793428916226,
            0.0660403820947893,
        ],
    )
    return layout_permissive_with_fractions


@pytest.fixture
def layout_restrictive():
    return Layout(
        [
            TextBlock(
                block=Rectangle(
                    x_1=49.28050231933594,
                    y_1=550.286865234375,
                    x_2=296.1264343261719,
                    y_2=671.5205078125,
                ),
                text=None,
                id=None,
                type="Text",
                parent=None,
                next=None,
                score=0.9980237483978271,
            ),
            TextBlock(
                block=Rectangle(
                    x_1=52.02384948730469,
                    y_1=418.7289123535156,
                    x_2=293.8260803222656,
                    y_2=538.3638916015625,
                ),
                text=None,
                id=None,
                type="Text",
                parent=None,
                next=None,
                score=0.9966004490852356,
            ),
            TextBlock(
                block=Rectangle(
                    x_1=68.77288818359375,
                    y_1=321.8116455078125,
                    x_2=291.71917724609375,
                    y_2=406.6871643066406,
                ),
                text=None,
                id=None,
                type="List",
                parent=None,
                next=None,
                score=0.9963681697845459,
            ),
            TextBlock(
                block=Rectangle(
                    x_1=50.793052673339844,
                    y_1=274.1769104003906,
                    x_2=280.36114501953125,
                    y_2=311.2925109863281,
                ),
                text=None,
                id=None,
                type="Text",
                parent=None,
                next=None,
                score=0.9950809478759766,
            ),
            TextBlock(
                block=Rectangle(
                    x_1=313.54071044921875,
                    y_1=655.3822631835938,
                    x_2=557.5581665039062,
                    y_2=715.7651977539062,
                ),
                text=None,
                id=None,
                type="Text",
                parent=None,
                next=None,
                score=0.993248462677002,
            ),
            TextBlock(
                block=Rectangle(
                    x_1=313.5169372558594,
                    y_1=241.96792602539062,
                    x_2=550.9529418945312,
                    y_2=326.1679992675781,
                ),
                text=None,
                id=None,
                type="Text",
                parent=None,
                next=None,
                score=0.9931153655052185,
            ),
            TextBlock(
                block=Rectangle(
                    x_1=312.76959228515625,
                    y_1=336.6008605957031,
                    x_2=544.313720703125,
                    y_2=420.7288818359375,
                ),
                text=None,
                id=None,
                type="Text",
                parent=None,
                next=None,
                score=0.9904166460037231,
            ),
            TextBlock(
                block=Rectangle(
                    x_1=313.13336181640625,
                    y_1=629.788818359375,
                    x_2=527.5404663085938,
                    y_2=640.5684204101562,
                ),
                text=None,
                id=None,
                type="Text",
                parent=None,
                next=None,
                score=0.9564889669418335,
            ),
            TextBlock(
                block=Rectangle(
                    x_1=311.0723571777344,
                    y_1=443.7219543457031,
                    x_2=563.8635864257812,
                    y_2=616.3117065429688,
                ),
                text=None,
                id=None,
                type="Table",
                parent=None,
                next=None,
                score=0.9412224292755127,
            ),
            TextBlock(
                block=Rectangle(
                    x_1=48.8629150390625,
                    y_1=239.5301055908203,
                    x_2=254.34307861328125,
                    y_2=261.0882873535156,
                ),
                text=None,
                id=None,
                type="Title",
                parent=None,
                next=None,
                score=0.9384030103683472,
            ),
            TextBlock(
                block=Rectangle(
                    x_1=54.44140625,
                    y_1=103.52156066894531,
                    x_2=372.6820373535156,
                    y_2=195.99090576171875,
                ),
                text=None,
                id=None,
                type="Title",
                parent=None,
                next=None,
                score=0.638062059879303,
            ),
        ]
    )


def test_split_layout(test_layout):
    layout_restrictive, layout_permissive = split_layout(test_layout, 0.5)
    # assert all scores in layout_restrictive are above 0.5 and all scores in layout_permissive are below 0.5
    assert all([b.score > 0.5 for b in layout_restrictive])
    assert all([b.score < 0.5 for b in layout_permissive])


@pytest.mark.parametrize(
    "test_input,expected", [((1, 2, 3, 4), Polygon([(1, 2), (1, 4), (3, 4), (3, 2)]))]
)
def test_lp_coords_to_shapely_polygon(test_input, expected):
    assert lp_coords_to_shapely_polygon(test_input) == expected


def test_calculate_unexplained_fractions(layout_permissive, layout_restrictive):
    permissive_layout_with_fractions = calculate_unexplained_fractions(
        layout_restrictive, layout_permissive
    )
    fractions = permissive_layout_with_fractions.unexplained_fractions
    # must be a list of floats between 0 and 1
    assert all([0 <= f <= 1 for f in fractions])
    # must be the same length as the number of blocks
    assert len(fractions) == len(layout_permissive)


def test_unnest_boxes(test_layout):
    # Setup.
    pixel_margin = 15
    soft_margin = {
        "top": pixel_margin,
        "bottom": pixel_margin,
        "left": pixel_margin,
        "right": pixel_margin,
    }
    unnested_layout = remove_nested_boxes(test_layout, pixel_margin)
    # Make sure no box is within another box (soft margin).
    for box in unnested_layout:
        for other_box in unnested_layout:
            if box != other_box:
                assert not box.is_in(other_box, soft_margin)
