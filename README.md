# Navigator Document Parser

Parsers for web pages and PDFs containing laws and policies.
## Setup

- `make install` - install dependencies using Poetry and set up playwright and pre-commit

## Running the CLI

To run in Docker with the `data/raw` folder as input and `data/processed` as output run:

``` bash
make run_docker
```

The CLI operates on an input folder of tasks defined by JSON files in the following format.

``` python
{
  "id": "test_html",
  "url": "https://www.industry.gov.au/funding-and-incentives/emissions-reduction-fund",
  "content_type": "text/html", # or "application/pdf"
  "document_slug": "YYY"
}
```

It outputs JSON files named `id.json`, with `id` being the ID of each input document, to an output folder in one of these formats:

``` python
# HTMLs
{
    "id": "test_html",
    "url": "https://www.industry.gov.au/funding-and-incentives/emissions-reduction-fund",
    "languages": [
        "en"
    ],
    "translated": false,
    "document_slug": "YYY",
    "content_type": "text/html",
    "html_data": {
        "detected_title": "Can't find what you are looking for?",
        "detected_date": "2022-05-15",
        "has_valid_text": false,
        "text_blocks": [
            {
                "text": [
                    "Can't find what you are looking for?"
                ],
                "text_block_id": "b0",
                "language": null,
                "type": "Text",
                "type_confidence": 1.0
            },
            {
                "text": [
                    "It looks like the page or file you are trying to access has moved, or the web address you have entered is incorrect."
                ],
                "text_block_id": "b1",
                "language": null,
                "type": "Text",
                "type_confidence": 1.0
            },
            {
                "text": [
                    "You can try:"
                ],
                "text_block_id": "b2",
                "language": null,
                "type": "Text",
                "type_confidence": 1.0
            }
        ]
    },
    "pdf_data": null
}

# PDFs
{
    "id": "test_pdf",
    "url": "https://cdn.climatepolicyradar.org/EUR/2013/EUR-2013-01-01-Overview+of+CAP+Reform+2014-2020_6237180d8c443d72c06c9167019ca177.pdf",
    "languages": null,
    "translated": false,
    "document_slug": "XYX",
    "content_type": "application/pdf",
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
                    596.0,
                    842.0
                ]
            }
        ],
        "md5sum": "6237180d8c443d72c06c9167019ca177",
        "text_blocks": [
            {
                "text": [
                    "Contact: 06 Agriculture, and\nural Development Unit Tor\nAgree Paley Anais\n"
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
                    "In short, EU agriculture needs to attain\nhigher levels of production of safe and\nquality food, while preserving the natural\nresources that agricultural productivity\ndepends upon.\n"
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