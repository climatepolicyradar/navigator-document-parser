from pathlib import Path
import json
import logging

import pandas as pd
from tqdm.auto import tqdm
import click

from src.trafilatura import TrafilaturaParser
from src.readability import ReadabilityParser
from src.newsplease import NewsPleaseParser
from src.combined import CombinedParser

logger = logging.getLogger(__name__)
logging.basicConfig(filename="parsing.log", filemode="w", level=logging.INFO)


@click.command()
@click.argument(
    "input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument(
    "output_folder", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
def main(input_path: Path, output_folder: Path):
    parsers = [
        TrafilaturaParser(),
        ReadabilityParser(),
        NewsPleaseParser(),
        CombinedParser(max_paragraph_words=500),
    ]
    sample_data = pd.read_csv(input_path)

    for parser in parsers:
        logger.info(f"Running for {parser.name}")
        for idx, row in tqdm(sample_data.iterrows(), total=len(sample_data)):
            output_data = parser.parse(row["url"]).detect_language()
            output_dict = output_data.dict()

            output_path = output_folder / f"{parser.name}_{row['id']}.json"

            with open(output_path, "w") as f:
                # `default=str` used to serialise date object
                # `ensure_ascii=False` used to create more readable JSON files. # TODO: we might not want to do this in the production pipeline
                f.write(
                    json.dumps(output_dict, default=str, ensure_ascii=False, indent=4)
                )


if __name__ == "__main__":
    main()
