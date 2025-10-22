import sys
import logging


def get_logger(log_file_path: str="training.log") -> logging.RootLogger:
    """
    Return a **root logger** object to record information in a log file.
    Use logging.info() anywhere to record information, since this is equivalent to logger.info() when logger is the root logger.

    Args:
        log_file_path (str): The path of the log file, where logs will be saved.

    Returns:
        logger (logging.RootLogger): A logger object to record information in a log file.

    Example Usage:
    ```
    logger = get_logger('log.log') # Get the root logger
    logging.debug("This is a debug message")
    logging.info("This is an info message")
    logging.error("This is an error message")
    logging.warning("This is a warning message")
    logging.critical("This is a critical message")
    ```
    """
    # Create the root logger object
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create and set a FileHandler (output to file)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)

    # Create and set a StreamHandler (output to console)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Create Formatter
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s", # log format
        datefmt="%Y-%m-%d %H:%M:%S" # time format
    )

    # bind the formatter to the handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def close_logger(logger: logging.Logger):
    """Close loggers and free up resources (If not, handlers in different runs can overlap.)"""
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
