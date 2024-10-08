import logging
from datetime import datetime, timezone
from typing import Optional

from aioxmpp import JID
from spade.template import Template

from .csv import CsvLogManager


class MessageLogManager(CsvLogManager):

    def __init__(
        self,
        base_logger_name="rf.message",
        extra_logger_name=None,
        level=logging.DEBUG,
        datetime_format="%Y-%m-%dT%H:%M:%S.%fZ",
        mode="a",
        encoding=None,
        delay=False,
    ):
        super().__init__(
            base_logger_name,
            extra_logger_name,
            level,
            datetime_format,
            mode,
            encoding,
            delay,
        )

    @staticmethod
    def get_header() -> str:
        return "log_timestamp,log_name,iteration_id,timestamp,sender,to,type,size"

    @staticmethod
    def get_template() -> Template:
        return Template(metadata={"rf.observer.log": "message"})

    def log(
        self,
        iteration_id: int,
        sender: str | JID,
        to: str | JID,
        msg_type: str,
        size: int,
        timestamp: Optional[datetime] = None,
        level: Optional[int] = None,
    ) -> None:
        lvl = self.level if level is None else level
        dt = datetime.now(tz=timezone.utc) if timestamp is None else timestamp
        dt_str = dt.strftime(self.datetime_format)
        sender = str(sender.bare()) if isinstance(sender, JID) else sender
        to = str(to.bare()) if isinstance(to, JID) else to
        msg = ",".join(
            [
                str(iteration_id),
                dt_str,
                sender,
                to,
                msg_type,
                str(size),
            ]
        )
        self.logger.log(level=lvl, msg=msg)
