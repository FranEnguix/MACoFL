import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from aioxmpp import JID

from .csv_handler import CsvFileHandler


class MessageLogManager:

    def __init__(
        self,
        log_folder: Path,
        base_logger_name: str = "rf.message",
        extra_logger_name: Optional[str] = None,
        level: int = logging.DEBUG,
        datetime_format: str = "",
        mode: str = "a",
        encoding: Optional[str] = None,
        delay: bool = False,
    ) -> None:
        self.log_path = log_folder / "message.csv"
        self.base_logger_name = base_logger_name
        self.extra_logger_name = extra_logger_name
        self.logger_name = (
            base_logger_name
            if extra_logger_name is None
            else f"{base_logger_name}.{extra_logger_name}"
        )
        self.level = level
        self.datetime_format = datetime_format
        self.mode = mode
        self.encoding = encoding
        self.delay = delay
        self.formatter = logging.Formatter(
            "%(asctime)s.%(msecs)03d,%(name)s,%(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        self.logger = logging.getLogger(self.logger_name)
        self.is_set_up = False

    def setup(self) -> None:
        if not self.log_path.exists():
            self.log_path.mkdir(parents=True, exist_ok=True)
        if not self.is_set_up:
            csv_handler = CsvFileHandler(
                path=self.log_path,
                header=self.get_header(),
                mode=self.mode,
                encoding=self.encoding,
                delay=self.delay,
            )
            csv_handler.setLevel(self.level)
            csv_handler.setFormatter(self.formatter)
            logger = logging.getLogger(self.base_logger_name)
            logger.setLevel(self.level)
            logger.addHandler(csv_handler)
        self.is_set_up = True

    def log(
        self,
        iteration_number: int,
        sender: JID,
        dest: JID,
        msg_type: str,
        size: int,
        timestamp: Optional[datetime] = None,
        level: Optional[int] = None,
    ) -> None:
        lvl = self.level if level is None else level
        dt = datetime.now() if timestamp is None else timestamp
        dt_str = dt.strftime(self.datetime_format)
        msg = ",".join(
            [
                str(iteration_number),
                dt_str,
                str(sender.bare()),
                str(dest.bare()),
                msg_type,
                str(size),
            ]
        )
        self.logger.log(level=lvl, msg=msg)

    @staticmethod
    def get_header() -> str:
        return "log_timestamp,log_name,iteration_number,timestamp,sender,dest,type,size"
