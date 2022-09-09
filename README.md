# HTML Parsing

Using article extraction techniques to parse web pages containing climate laws and policies.
## Setup

- `make install` - install dependencies using Poetry and set up playwright and pre-commit

## Running the CLI

The CLI operates on an input folder of tasks defined by JSON files in the following format.

``` json
{
    "id": "document_id",
    "url": "https://www.gov.uk/feed-in-tariffs"
}
```

It outputs JSON files named `id.json`, with `id` being the ID of each input documebt, to an output folder in this format:

``` json
{
    "id": "test_id",
    "url": "https://www.gov.uk/feed-in-tariffs",
    "title": "Smart Export Guarantee (SEG): earn money for exporting the renewable electricity you have generated",
    "text_by_line": [
        "If you generate renewable electricity in your home or business, you can feed back into the grid any electricity that you don’t use. Under the Smart Export Guarantee ( SEG ) you will be paid for every unit of electricity that you feed back. You won’t be paid for any that you use yourself.",
        "What you need to apply for a SEG tariff",
        "You need to have a renewable electricity generating system that meets the SEG eligibility requirements.",
        "You must have a meter capable of providing half-hourly export readings. This would typically be a smart meter. Speak to your energy supplier about getting a smart meter installed if you do not already have one.",
        "You need to show that your installation and installer are certified through the microgeneration certification scheme (MCS) or equivalent.",
        "You cannot receive a SEG tariff if you are receiving export payments under the Feed-in Tariff scheme.",
        "How to get a SEG tariff",
        "You need to apply directly to a SEG tariff supplier to get paid. The Ofgem website lists the energy suppliers that provide SEG tariffs.",
        "Your SEG tariff supplier does not need to be the same as the supplier that provides your energy.",
        "SEG suppliers are required to offer you a tariff but are free to determine the terms of the tariff they offer, for example whether it is fixed or variable.",
        "Tariffs can change over time so you should regularly check to make sure you remain on a competitive tariff.",
        "If you have a storage device, such as a household battery or electric vehicle, that has the ability to import and export electricity, it could also be used to benefit from the SEG . Your prospective SEG tariff supplier can advise you about this.",
        "How much could you save?",
        "Use the Energy Saving Trust calculator to estimate:",
        "how much you could save from solar panels or other renewable electricity generating systems",
        "how much you could earn selling unused energy back",
        "Although you will not be paid for electricity that you use yourself, you will save money through importing less from the grid."
    ],
    "date": "2020-01-01",
    "has_valid_text": true,
    "language": "en",
    "translated": false
}
```