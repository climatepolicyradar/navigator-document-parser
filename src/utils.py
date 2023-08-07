from urllib.parse import urlencode

from src.config import SCRAPEOPS_API_KEY


def get_proxy_url(url: str) -> str:
    """Get a ScrapeOps proxy URL for the given URL."""

    if not SCRAPEOPS_API_KEY:
        raise ValueError(
            "SCRAPEOPS_API_KEY must be set in environment variables to use a scrapeops "
            "proxy. "
        )

    payload = {"api_key": SCRAPEOPS_API_KEY, "url": url}

    return "https://proxy.scrapeops.io/v1/?" + urlencode(payload)
