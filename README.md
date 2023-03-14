# Navigator Document Parser

Parsers for web pages and PDFs containing laws and policies.

Please see config.py for configuration options and their explanations. There are many hyperparameters that can be tuned to improve the performance of the parser.
## Setup

- `make install` - install dependencies using Poetry and set up playwright and pre-commit

To run locally you will then need to enter your Poetry virtual environment with `poetry shell`.

## Running the CLI

To run in Docker with the `data/raw` folder as input and `data/processed` as output run:

``` bash
make run_docker
```

The CLI operates on an input folder of tasks defined by JSON files in the following format.

``` python
{
  "document_id": "test_html",
  "document_metadata": {},
  "document_name": "test_html",
  "document_description": "test_html description",
  "document_url": "https://www.industry.gov.au/funding-and-incentives/emissions-reduction-fund",
  "document_content_type": "text/html", # or "application/pdf"
  "document_slug": "YYY"
}
```

It outputs JSON files named `id.json`, with `id` being the ID of each input document, to an output folder in one of these formats:

``` python
# HTMLs
{
    "document_id": "1",
    "document_name": "Policy Document 1",
    "document_url": "https://website.org/path/document",
    "document_metadata": {}, 
    "languages": [
        "en"
    ],
    "translated": false,
    "document_slug": "YYYY",
    "document_content_type": "text/html",
    "html_data": {
        "detected_title": "[no-title]",
        "detected_date": null,
        "has_valid_text": true,
        "text_blocks": [
            {
                "text": [
                    "%PDF-1.6"
                ],
                "text_block_id": "b0",
                "language": "da",
                "type": "Text",
                "type_confidence": 1.0
            }
        ]
    },
    "pdf_data": null
}

# PDFs
{
    "document_id": "1",
    "document_name": "Policy Document 1",
    "document_url": "https://website.org/path/document",
    "languages": null,
    "translated": false,
    "document_slug": "YYYY",
    "document_content_type": "application/pdf",
    "html_data": null,
    "pdf_data": {
        "page_metadata": [
            {
                "page_number": 0,
                "dimensions": [
                    596.0,
                    842.0
                ]
            },
            {
                "page_number": 1,
                "dimensions": [
                    612.0,
                    936.0
                ]
            },
            {
                "page_number": 2,
                "dimensions": [
                    612.0,
                    936.0
                ]
            },
            {
                "page_number": 3,
                "dimensions": [
                    612.0,
                    936.0
                ]
            },
            {
                "page_number": 4,
                "dimensions": [
                    612.0,
                    936.0
                ]
            }
        ],
        "md5sum": "6237180d8c443d72c06c9167019ca177",
        "text_blocks": [
            {
                "text": [
                    Example text block."
                ],
                "text_block_id": "p_0_b_0",
                "language": null,
                "type": "Text",
                "type_confidence": 0.6339805126190186,
                "coords": [
                    [
                        10.998469352722168,
                        702.727294921875
                    ],
                    [
                        134.93479919433594,
                        702.727294921875
                    ],
                    [
                        134.93479919433594,
                        737.7978515625
                    ],
                    [
                        10.998469352722168,
                        737.7978515625
                    ]
                ],
                "page_number": 0
            },
            {
                "text": [
                    "Example text block."
                ],
                "text_block_id": "p_1_b_0",
                "language": null,
                "type": "Title",
                "type_confidence": 0.577865481376648,
                "coords": [
                    [
                        26.7734375,
                        313.9053039550781
                    ],
                    [
                        281.1876525878906,
                        313.9053039550781
                    ],
                    [
                        281.1876525878906,
                        380.9349365234375
                    ],
                    [
                        26.7734375,
                        380.9349365234375
                    ]
                ],
                "page_number": 1
            },
        ]
    }
}
```