import logging
from pythonjsonlogger import jsonlogger
from datetime import datetime
from src.config import LOGGING_LEVEL


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter for logging.

    This class is used to format the logs in JSON format.
    """

    def add_fields(self, log_record, record, message_dict):
        """Add fields to the log record.

        This method is used to add fields to the log record.
        """
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        if not log_record.get("timestamp"):
            # this doesn't use record.created, so it is slightly off
            now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            log_record["timestamp"] = now
        if log_record.get("level"):
            log_record["level"] = log_record["level"].upper()
        else:
            log_record["level"] = record.levelname
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        if not log_record.get("timestamp"):
            # this doesn't use record.created, so it is slightly off
            now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            log_record["timestamp"] = now
        if log_record.get("level"):
            log_record["level"] = log_record["level"].upper()
        else:
            log_record["level"] = record.levelname


def get_logger(name):
    logger = logging.getLogger(name)
    if LOGGING_LEVEL == "DEBUG":
        logger.setLevel(logging.DEBUG)
    elif LOGGING_LEVEL == "WARNING":
        logger.setLevel(logging.WARNING)
    elif LOGGING_LEVEL == "ERROR":
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)
    logHandler = logging.StreamHandler()
    formatter = CustomJsonFormatter("%(timestamp)s %(level)s %(name)s %(message)s")
    logHandler.setFormatter(formatter)
    logger.addHandler(logHandler)
    return logger
