import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


class CsvFileHandler(logging.FileHandler):
    def __init__(
        self,
        filename: Path,
        header: str,
        mode: str = "a",
        encoding: Optional[str] = None,
        delay: bool = False,
    ):
        self.header = header
        file_exists = os.path.exists(filename)
        super().__init__(filename, mode=mode, encoding=encoding, delay=delay)
        if not file_exists or mode == "w" or os.path.getsize(filename) == 0:
            if not self.stream:
                self.stream = self._open()
            self.stream.write(self.header + "\n")
            self.stream.flush()


def _setup_csv_log(
    filename: Path,
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
    log_folder_path: str | Path = "logs",
    datetime_mark: bool = True,
    general_level: int = logging.DEBUG,
    csv_level: int = logging.DEBUG,
) -> None:
    log_folder = (
        Path(log_folder_path) if isinstance(log_folder_path, str) else log_folder_path
    )
    if datetime_mark:
        log_folder = log_folder / datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    log_folder.mkdir(parents=True, exist_ok=True)

    # Set up the base format and log levels for the different loggers
    formatter = logging.Formatter(
        "%(asctime)s, %(name)s, %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    csv_formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d,%(name)s,%(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # General agent log
    rf_console_log_handler = logging.StreamHandler()
    rf_console_log_handler.setLevel(general_level)
    rf_console_log_handler.setFormatter(formatter)

    rf_log_path = log_folder / "general.log"
    rf_log_handler = logging.FileHandler(rf_log_path)
    rf_log_handler.setLevel(general_level)
    rf_log_handler.setFormatter(formatter)

    # Attach handlers to the loggers for each agent category
    base_logger = logging.getLogger("rf.log")
    base_logger.setLevel(general_level)
    base_logger.addHandler(rf_log_handler)
    base_logger.addHandler(rf_console_log_handler)

    # Filehandlers logs
    csv_logs = [
        (
            "message.csv",
            "rf.message",
            "log_timestamp,log_name,iteration_number,timestamp,sender,dest,type,size",
        ),
        (
            "nn.csv",
            "rf.nn",
            "log_timestamp,log_name,iteration_number,timestamp,agent,seconds_to_complete,training_accuracy,training_loss,test_accuracy,test_loss",
        ),
        (
            "iteration.csv",
            "rf.iteration",
            "log_timestamp,log_name,iteration_number,timestamp,agent,seconds_to_complete",
        ),
    ]
    for filename, logger_name, header in csv_logs:
        csv_path = log_folder / filename
        _setup_csv_log(csv_path, logger_name, csv_level, csv_formatter, header)
