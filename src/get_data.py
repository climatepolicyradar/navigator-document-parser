"""Create sample of links to HTML pages to test parsing approaches."""

from pathlib import Path

import pandas as pd
import click
import numpy as np


@click.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_file", type=click.Path(dir_okay=False))
def main(input_file: str, output_file: str) -> None:
    """Create a sample of links to HTML pages from the prototype CSV, stratified by language.

    Arguments:
        input_file -- path to prototype `processed_policies.csv` file
        output_file -- path to csv file to write sample to
    """
    RANDOM_STATE = 42

    df = pd.read_csv(input_file, index_col=0)

    df = df[(df["doc_mime_type"] == "text/html") & (df["source_name"] == "cclw")]

    df["id"] = df["policy_txt_file"].apply(
        lambda i: Path(i).stem if pd.notnull(i) else np.nan
    )

    stratified_sample_by_language = (
        df.dropna(subset=["policy_txt_file"])
        .groupby("language", group_keys=False)
        .apply(lambda x: x.sample(frac=0.2, random_state=RANDOM_STATE))[
            [
                "id",
                "policy_name",
                "country_code",
                "url",
                "policy_description",
                "policy_date",
            ]
        ]
    )  # type: ignore

    stratified_sample_by_language.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
