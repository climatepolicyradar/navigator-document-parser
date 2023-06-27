"""Module presents a method for downloading the model that is to be used in the parser.

This is so that it can be downloaded once and form a layer of the docker image rather than being downloaded at
run time every time the parser is instantiated.
"""

from src.pdf_parser.layout import LayoutParserWrapper


if __name__ == "__main__":
    lp_obj = LayoutParserWrapper()
