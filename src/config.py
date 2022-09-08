import os

MIN_NO_LINES_FOR_VALID_TEXT = int(os.getenv("MIN_NO_LINES_FOR_VALID_TEXT", "6"))
HTTP_REQUEST_TIMEOUT = int(os.getenv("HTTP_REQUEST_TIMEOUT", "30"))  # seconds

# TODO: http request headers?
