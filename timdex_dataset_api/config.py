import logging
import os


def configure_logger(
    name: str,
    warning_only_loggers: str | None = None,
) -> logging.Logger:
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

    warning_only_loggers = os.getenv("WARNING_ONLY_LOGGERS", warning_only_loggers)
    if warning_only_loggers:
        for warning_logger_name in warning_only_loggers.split(","):
            logging.getLogger(warning_logger_name).setLevel(logging.WARNING)

    return logger


def configure_dev_logger() -> logging.Logger:
    """Invoke to setup DEBUG level console logging for development work."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger("timdex_dataset_api")
    logger.setLevel(logging.DEBUG)
    return logger
