import logging
import os


def configure_logger(name: str) -> logging.Logger:
    """Prepares a logger instance.

    If the env var TDA_LOG_LEVEL is set, the logging level will override the logging
    level of the calling context.

    Args:
        name (str): The name of the logger, typically __name__ is passed by caller
    """
    logger = logging.getLogger(name)

    # set logger level if env var 'TDA_LOG_LEVEL' is set
    if log_level := os.getenv("TDA_LOG_LEVEL"):
        log_level = log_level.strip().upper()
        if log_level not in logging.getLevelNamesMapping():
            raise ValueError(f"Invalid log level: '{log_level}'")
        logger.setLevel(getattr(logging, log_level))

    return logger
