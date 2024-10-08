import logging
from pathlib import Path
from typing import Optional


class GeneralLogManager:

    def __init__(
        self,
        base_logger_name: str = "rf.log",
        extra_logger_name: Optional[str] = None,
        level: int = logging.DEBUG,
    ) -> None:
        self.base_logger_name = base_logger_name
        self.extra_logger_name = extra_logger_name
        self.level = level
        self.formatter = logging.Formatter(
            "%(asctime)s, %(name)s, %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        self.logger = self.__get_logger()
        self.is_set_up = False

    def __get_logger(self) -> logging.Logger:
        self.logger_name = (
            self.base_logger_name
            if self._extra_logger_name is None
            else f"{self.base_logger_name}.{self._extra_logger_name}"
        )
        return logging.getLogger(self.logger_name)

    @property
    def extra_logger_name(self) -> str | None:
        return self._extra_logger_name

    @extra_logger_name.setter
    def extra_logger_name(self, value: str) -> None:
        self._extra_logger_name = value
        self.__get_logger()

    def setup(self, folder_name: str | Path, file_name: str | Path) -> None:
        log_path = Path(folder_name)
        if not log_path.exists():
            log_path.mkdir(parents=True, exist_ok=True)
        log_path = Path(folder_name) / file_name
        logger = logging.getLogger(self.base_logger_name)
        if len(logger.handlers) == 0:
            # General agent log
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.level)
            console_handler.setFormatter(self.formatter)

            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(self.level)
            file_handler.setFormatter(self.formatter)

            # Attach handlers to the loggers for each agent category
            logger = logging.getLogger(self.base_logger_name)
            logger.setLevel(self.level)
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

    def debug(self, msg: object) -> None:
        self.logger.debug(msg)

    def info(self, msg: object) -> None:
        self.logger.info(msg)

    def error(self, msg: object) -> None:
        self.logger.error(msg)

    def exception(self, msg: object) -> None:
        self.logger.exception(msg)
