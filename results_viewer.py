from pathlib import Path

import streamlit as st
import pandas as pd

from src.base import ParsedHTML


def load_data() -> pd.DataFrame:
    data_sample_path = "./data/interim/html_sample.csv"
    return pd.read_csv(data_sample_path)


def get_parser_results(page_id, parser_name) -> ParsedHTML:
    results_folder = Path("./data/processed")

    path = results_folder / f"{parser_name}_{page_id}.json"

    return ParsedHTML.parse_file(path)


def display_parsed_html(tab, parsed_html: ParsedHTML):
    tab.markdown("### title")
    tab.write(parsed_html.title)
    tab.markdown("### valid text")
    tab.write(parsed_html.has_valid_text)
    tab.markdown("### language")
    tab.write(parsed_html.language)
    tab.markdown("### text")
    tab.write("\n\n".join(parsed_html.text_by_line))


if __name__ == "__main__":
    data_sample = load_data()

    st.header("Results Viewer")

    CURRENT_INDEX = st.number_input("page number", 0, len(data_sample) - 1, 0)

    current_id = data_sample.iloc[CURRENT_INDEX]["id"]

    st.write(data_sample.iloc[CURRENT_INDEX]["url"])

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Trafilatura", "Readability", "NewsPlease", "Combined"]
    )

    trafilatura_results = get_parser_results(current_id, "trafilatura")
    display_parsed_html(tab1, trafilatura_results)

    readability_results = get_parser_results(current_id, "readability")
    display_parsed_html(tab2, readability_results)

    newsplease_results = get_parser_results(current_id, "newsplease")
    display_parsed_html(tab3, newsplease_results)

    combined_results = get_parser_results(current_id, "combined")
    display_parsed_html(tab4, combined_results)
