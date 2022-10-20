from pathlib import Path
from typing import Union

from cloudpathlib import CloudPath
from tqdm.auto import tqdm

from src.config import TARGET_LANGUAGES  # noqa: E402
from src.base import ParserOutput, LogProps  # noqa: E402
from src.translator.translate import translate_parser_output  # noqa: E402
from src.utils import get_logger
from src.config import PIPELINE_STAGE  # noqa: E402
from src.config import PIPELINE_RUN  # noqa: E402

logger = get_logger(__name__)
default_extras = {
    "props": LogProps.parse_obj(
        {
            "pipeline_run": PIPELINE_RUN,
            "pipeline_stage": PIPELINE_STAGE,
            "pipeline_stage_subsection": f"{__name__}",
            "document_in_process": None,
            "error": None,
        }
    ).dict()
}


def translate_parser_outputs(parser_output_dir: Union[Path, CloudPath]) -> None:
    """
    Translate parser outputs saved in the output directory, and save the translated outputs to the output directory.

    :param parser_output_dir: directory containing parser outputs
    """
    for path in tqdm(parser_output_dir.glob("*.json")):

        parser_output = ParserOutput.parse_raw(path.read_text())

        # Skip already translated outputs. Note this does not prevent the CLI from translating existing parser
        # outputs again, but instead makes sure it doesn't translate a translation.
        if parser_output.translated:
            continue

        _target_languages = set(TARGET_LANGUAGES)

        # If there is only one language that's been detected in the parser output, we don't need to translate to this
        # language. TODO: how do we deal with the fact that parser outputs can contain multiple languages here?
        if parser_output.languages and len(parser_output.languages) == 1:
            _target_languages = _target_languages - set(parser_output.languages)

        for target_language in _target_languages:
            output_path = path.with_name(
                f"{path.stem}_translated_{target_language}.json"
            )

            if output_path.exists():
                logger.info(
                    f"Skipping translating {output_path} because it already exists.",
                    extra={
                        "props": LogProps.parse_obj(
                            {
                                "pipeline_run": PIPELINE_RUN,
                                "pipeline_stage": PIPELINE_STAGE,
                                "pipeline_stage_subsection": f"{__name__}",
                                "document_in_process": f"{output_path}",
                                "error": None,
                            }
                        ).dict()
                    },
                )
                continue

            translated_parser_output = translate_parser_output(
                parser_output, target_language
            )

            output_path.write_text(
                translated_parser_output.json(indent=4, ensure_ascii=False)
            )
