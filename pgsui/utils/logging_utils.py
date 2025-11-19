import logging


def configure_logger(
    logger: logging.Logger,
    *,
    verbose: bool = False,
    debug: bool = False,
    quiet_level: int = logging.ERROR,
) -> logging.Logger:
    """Force a logger and its handlers to respect verbose/debug controls."""
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = quiet_level

    logger.setLevel(level)
    for handler in getattr(logger, "handlers", ()):
        handler.setLevel(level)
    return logger
