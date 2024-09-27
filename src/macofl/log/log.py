import logging

from typing import Optional


def _setup_filehandler_log(
    filename: str, logger_name: str, level: Optional[int], formatter: logging.Formatter
) -> None:
    log_handler = logging.FileHandler(filename)
    log_handler.setLevel(level)
    log_handler.setFormatter(formatter)
    log = logging.getLogger(logger_name)
    log.setLevel(level)
    log.addHandler(log_handler)


def setup_loggers(
    general_level: Optional[int] = logging.DEBUG,
    filehandler_level: Optional[int] = logging.INFO,
) -> None:
    # Set up the base format and log levels for the different loggers
    formatter = logging.Formatter(
        "%(asctime)s; %(name)s; %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # General agent log
    rf_console_log_handler = logging.StreamHandler()
    rf_console_log_handler.setLevel(general_level)
    rf_console_log_handler.setFormatter(formatter)

    rf_log_handler = logging.FileHandler("general.log")
    rf_log_handler.setLevel(general_level)
    rf_log_handler.setFormatter(formatter)

    # Attach handlers to the loggers for each agent category
    base_logger = logging.getLogger("rf.log")
    base_logger.setLevel(general_level)
    base_logger.addHandler(rf_log_handler)
    base_logger.addHandler(rf_console_log_handler)

    # Filehandlers logs
    filehandler_logs = [
        ("message.log", "rf.message"),
        ("accuracy.log", "rf.accuracy"),
        ("loss.log", "rf.loss"),
        ("iteration.log", "rf.iteration"),
    ]
    for filename, logger_name in filehandler_logs:
        _setup_filehandler_log(filename, logger_name, filehandler_level, formatter)
