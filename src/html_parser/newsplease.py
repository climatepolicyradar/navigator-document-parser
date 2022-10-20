"""Parser using news-please library: https://github.com/fhamborg/news-please"""


from newsplease import NewsPlease
import requests

from src.base import (
    HTMLParser,
    ParserInput,
    ParserOutput,
    HTMLTextBlock,
    HTMLData,
    LogProps,
    ErrorLog,
)
from src.config import HTML_MIN_NO_LINES_FOR_VALID_TEXT, HTML_HTTP_REQUEST_TIMEOUT
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


class NewsPleaseParser(HTMLParser):
    """HTML parser which uses the news-please library."""

    def __init__(self) -> None:
        super().__init__()

    @property
    def name(self) -> str:
        """Return parser name"""
        return "newsplease"

    def parse_html(self, html: str, input: ParserInput) -> ParserOutput:
        """
        Parse HTML using newsplease.

        :param html: HTML string to parse
        :param url: URL of web page

        :return ParsedHTML: parsed HTML
        """

        try:
            article = NewsPlease.from_html(  # pyright: ignore
                html=html, url=input.document_url, fetch_images=False
            )
        except Exception as e:
            logger.error(
                f"Failed to parse {input.document_url} for {input.document_id}: {e}",
                extra=default_extras,
            )
            return self._get_empty_response(input)

        return self._newsplease_article_to_parsed_html(article, input)

    def parse(self, input: ParserInput) -> ParserOutput:
        """
        Parse website using newsplease

        :param url: URL of web page

        :return ParsedHTML: parsed HTML
        :raise ValueError: input contains a null value for URL
        """

        if input.document_url is None:
            raise ValueError(
                "A URL is required, and it seems like a document without a URL was provided."
            )

        try:
            response = requests.get(
                input.document_url,
                verify=False,
                allow_redirects=True,
                timeout=HTML_HTTP_REQUEST_TIMEOUT,
            )

        except Exception as e:
            logger.error(
                "Error downloading file; skipping...",
                extra={
                    "props": LogProps.parse_obj(
                        {
                            "pipeline_run": PIPELINE_RUN,
                            "pipeline_stage": PIPELINE_STAGE,
                            "pipeline_stage_subsection": f"{__name__} - requests.get",
                            "document_in_process": f"{input.document_id}",
                            "error": ErrorLog.parse_obj(
                                {"status_code": None, "error_message": f"{e}"}
                            ),
                        }
                    ).dict()
                },
            )
            return self._get_empty_response(input)

        return self.parse_html(response.text, input)

    def _newsplease_article_to_parsed_html(
        self, newsplease_article, input: ParserInput
    ) -> ParserOutput:
        """
        Convert a newsplease article to parsed HTML. Returns an empty response if the article contains no text.

        :param newsplease_article: article returned by `NewsPlease.from_url` or `NewsPlease.from_html`
        :param url: URL of web page

        :return ParsedHTML: parsed HTML
        """

        text = newsplease_article.maintext

        if not text:
            return self._get_empty_response(input)

        text_by_line = text.split("\n")
        has_valid_text = len(text_by_line) >= HTML_MIN_NO_LINES_FOR_VALID_TEXT

        text_blocks = [
            HTMLTextBlock.parse_obj(
                {
                    "text_block_id": f"b{idx}",
                    "text": [text],
                }
            )
            for idx, text in enumerate(text_by_line)
        ]

        return ParserOutput(
            document_id=input.document_id,
            document_metadata=input.document_metadata,
            document_url=input.document_url,
            document_name=input.document_name,
            document_description=input.document_description,
            document_content_type=input.document_content_type,
            document_slug=input.document_slug,
            html_data=HTMLData(
                detected_title=newsplease_article.title,
                detected_date=newsplease_article.date_publish,  # We also have access to the modified and downloaded dates in the class
                has_valid_text=has_valid_text,
                text_blocks=text_blocks,
            ),
        )
