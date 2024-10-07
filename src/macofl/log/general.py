import logging
from pathlib import Path
from typing import Optional


class GeneralLogManager:

    def __init__(
        self,
        log_folder: Path,
        base_logger_name: str = "rf.log",
        extra_logger_name: Optional[str] = None,
        level: int = logging.DEBUG,
    ) -> None:
        self.log_path = log_folder / "general.log"
        self.base_logger_name = base_logger_name
        self.extra_logger_name = extra_logger_name
        self.logger_name = (
            base_logger_name
            if extra_logger_name is None
            else f"{base_logger_name}.{extra_logger_name}"
        )
        self.level = level
        self.formatter = logging.Formatter(
            "%(asctime)s, %(name)s, %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        self.logger = logging.getLogger(self.logger_name)
        self.is_set_up = False

    def setup(self) -> None:
        if not self.log_path.exists():
            self.log_path.mkdir(parents=True, exist_ok=True)
        if not self.is_set_up:
            # General agent log
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.level)
            console_handler.setFormatter(self.formatter)

            file_handler = logging.FileHandler(self.log_path)
            file_handler.setLevel(self.level)
            file_handler.setFormatter(self.formatter)

            # Attach handlers to the loggers for each agent category
            logger = logging.getLogger(self.base_logger_name)
            logger.setLevel(self.level)
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        self.is_set_up = True
