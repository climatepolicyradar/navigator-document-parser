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
  "document_url": "https://www.industry.gov.au/funding-and-incentives/emissions-reduction-fund",
  "document_content_type": "text/html", # or "application/pdf"
  "document_slug": "YYY"
}
```

It outputs JSON files named `id.json`, with `id` being the ID of each input document, to an output folder in one of these formats:

``` python
# HTMLs
{
    "document_id": "8864.0",
    "document_name": "Presidential Decree No. 29/16 approving the National Plan for the preparation, resilience, response and recovery from natural disasters 2015-2017",
    "document_description": "This Presidential Decree approves the National Plan for the preparation, resilience, response and recovery from natural disasters for the period 2015-2017. This National Plan, consisting of 6 Sections, containing maps and tables, establishes the disaster risks and the steps to be taken while a natural emergency is occurring over the National territory, including droughts, floods, fires, etc. This Strategic Plan aims to contribute to the country's sustainable development process, by reducing vulnerability to disaster.",
    "document_url": "https://climate-laws.org/rails/active_storage/blobs/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBcU1GIiwiZXhwIjpudWxsLCJwdXIiOiJibG9iX2lkIn19--f86541576724b026f60469bcc9ebb1b34113feb5/f|",
    "languages": [
        "da"
    ],
    "translated": false,
    "document_slug": "document_slug_28c8e5ed-7169-486f-acf9-72b148ce6a6b",
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
    "document_id": "1332.0",
    "document_name": "Presidential of the Republic of Indonesia Instruction Number 6 Year 2013 on Suspension of New Licenses and Improving Forest Governance of Primary Forest and Peatland",
    "document_description": "The first iteration of this instruction was issued in 2011 in order to implement commitments under the agreements in the Letter of Intent signed with the Kingdom of Norway in May 2011. The Instruction is intended to facilitate Indonesia's participation in internationally financed REDD activities and places a moratorium on clearance of primary peatland and forests within moratorium areas. The initial moratorium was extended by Presidential Instruction 6/2013. In 2019, President Joko Widodo signed Presidential Instruction 5/2019, making the moratorium on the clearance of primary forest and peatlands in moratorium areas permanent. ",
    "document_url": "https://climate-laws.org/rails/active_storage/blobs/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBaUVHIiwiZXhwIjpudWxsLCJwdXIiOiJibG9iX2lkIn19--2edc173ca6e688a1d121c90f6282721aca9b0ca7/f|id",
    "languages": null,
    "translated": false,
    "document_slug": "document_slug_53b61eea-b6de-4eff-9f47-74f1b10b30ac",
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
