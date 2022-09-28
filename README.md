# Navigator Document Parser

Parsers for web pages and PDFs containing laws and policies.
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
  "document_name": "Policy Document 1"
  "document_url": "https://www.website.gov.uk/policy_document",
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
    "languages": [
        "da"
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
                    612.0,
                    936.0
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
        "md5sum": "3e465fc3e780f4933960e339bd0e70f6",
        "text_blocks": [
            {
                "text": [
                    "menginstruksikan:\n\f"
                ],
                "text_block_id": "p_0_b_0",
                "language": null,
                "type": "Text",
                "type_confidence": 0.9591814875602722,
                "coords": [
                    [
                        71.69662475585938,
                        405.9636535644531
                    ],
                    [
                        181.8621826171875,
                        405.9636535644531
                    ],
                    [
                        181.8621826171875,
                        418.4075012207031
                    ],
                    [
                        71.69662475585938,
                        418.4075012207031
                    ]
                ],
                "page_number": 0
            },
            {
                "text": [
                    "Untuk:\n\f"
                ],
                "text_block_id": "p_0_b_1",
                "language": null,
                "type": "Title",
                "type_confidence": 0.7153777480125427,
                "coords": [
                    [
                        71.7964096069336,
                        692.2425537109375
                    ],
                    [
                        113.10400390625,
                        692.2425537109375
                    ],
                    [
                        113.10400390625,
                        706.488525390625
                    ],
                    [
                        71.7964096069336,
                        706.488525390625
                    ]
                ],
                "page_number": 0
            }
        ]
    }
}
```
