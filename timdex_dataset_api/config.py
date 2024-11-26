import logging
import os


def configure_logger(name: str) -> logging.Logger:
    """Prepares and returns a logger instance for a given module name.

    This approach is suitable for an installed and imported library such as this, where
    any calling application logging levels and handlers should be utilized.

    Args:
        name (str): The name of the logger, typically __name__ is passed by caller
    """
    logger = logging.getLogger(name)
    logger.addHandler(logging.NullHandler())

    log_level = os.getenv("TDA_LOG_LEVEL", "INFO").strip().upper()
    if log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        raise ValueError(f"Invalid log level: '{log_level}'")

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s.%(funcName)s() "
            "line %(lineno)d: %(message)s"
        )
    )
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, log_level))

    return logger
