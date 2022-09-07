Testing some article extraction approaches sourced from this Article Extraction Benchmark: https://github.com/scrapinghub/article-extraction-benchmark

## Testing / tested

- [x] [trafilatura](https://github.com/adbar/trafilatura)
- [x] [python-readability](https://github.com/buriy/python-readability)
- ~~[dragnet](https://github.com/dragnet-org/dragnet)~~ not maintained
- [x] [news-please](https://github.com/fhamborg/news-please)

## To run

- `make data`: make sample of html files for testing
- `make run_parsers`: run all parsers on sample html files (note: also runs `make data`)
- `streamlit run results_viewer.py`: start app to view parsing results side by side