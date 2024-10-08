import logging
from datetime import datetime, timezone
from typing import Optional

from aioxmpp import JID
from spade.template import Template

from .csv import CsvLogManager


class AlgorithmLogManager(CsvLogManager):

    def __init__(
        self,
        base_logger_name="rf.algorithm",
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
        return "log_timestamp,log_name,iteration_number,timestamp,agent,seconds_to_complete"

    @staticmethod
    def get_template() -> Template:
        return Template(metadata={"rf.observer.log": "algorithm"})

    def log(
        self,
        iteration_id: int,
        agent: JID,
        seconds: float,
        timestamp: Optional[datetime] = None,
        level: Optional[int] = None,
    ) -> None:
        lvl = self.level if level is None else level
        dt = datetime.now(tz=timezone.utc) if timestamp is None else timestamp
        dt_str = dt.strftime(self.datetime_format)
        agent = str(agent.bare()) if isinstance(agent, JID) else agent
        msg = ",".join(
            [
                str(iteration_id),
                dt_str,
                agent,
                str(seconds),
            ]
        )
        self.logger.log(level=lvl, msg=msg)
