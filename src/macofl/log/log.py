import logging
import os
from typing import Optional


class CsvFileHandler(logging.FileHandler):
    def __init__(
        self,
        filename: str,
        header: str,
        mode: str = "a",
        encoding: Optional[str] = None,
        delay: bool = False,
    ):
        self.header = header
        file_exists = os.path.exists(filename)
        super().__init__(filename, mode=mode, encoding=encoding, delay=delay)
        if not file_exists or mode == "w":
            if not self.stream:
                self.stream = self._open()
            self.stream.write(self.header + "\n")
            self.stream.flush()


def _setup_filehandler_log(
    filename: str,
    logger_name: str,
    level: int,
    formatter: logging.Formatter,
    header: str,
) -> None:
    log_handler = CsvFileHandler(filename, header)
    log_handler.setLevel(level)
    log_handler.setFormatter(formatter)
    log = logging.getLogger(logger_name)
    log.setLevel(level)
    log.addHandler(log_handler)


def setup_loggers(
    general_level: int = logging.DEBUG,
    filehandler_level: int = logging.DEBUG,
) -> None:
    # Set up the base format and log levels for the different loggers
    formatter = logging.Formatter(
        "%(asctime)s, %(name)s, %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    filehandler_formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d, %(name)s, %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
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
        (
            "message.log",
            "rf.message",
            "log_timestamp,log_name,iteration_id,timestamp,sender,dest,type,size",
        ),
        (
            "nn.log",
            "rf.nn",
            "log_timestamp,log_name,iteration_id,timestamp,agent,training_accuracy,training_loss,test_accuracy,test_loss",
        ),
        (
            "iteration.log",
            "rf.iteration",
            "log_timestamp,log_name,iteration_id,timestamp,agent,seconds",
        ),
    ]
    for filename, logger_name, header in filehandler_logs:
        _setup_filehandler_log(
            filename, logger_name, filehandler_level, filehandler_formatter, header
        )
