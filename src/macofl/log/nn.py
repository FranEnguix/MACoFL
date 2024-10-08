import logging
from datetime import datetime, timezone
from typing import Optional

from aioxmpp import JID
from spade.template import Template

from .csv import CsvLogManager


class NnLogManager(CsvLogManager):

    def __init__(
        self,
        base_logger_name="rf.nn",
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
        return "log_timestamp,log_name,iteration_number,timestamp,agent,seconds_to_complete,training_accuracy,training_loss,test_accuracy,test_loss"

    @staticmethod
    def get_template() -> Template:
        return Template(metadata={"rf.observer.log": "nn"})

    def log(
        self,
        iteration_id: int,
        agent: str | JID,
        seconds: float,
        training_accuracy: float,
        training_loss: float,
        test_accuracy: float,
        test_loss: float,
        timestamp: Optional[datetime] = None,
        level: Optional[int] = logging.DEBUG,
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
                str(training_accuracy),
                str(training_loss),
                str(test_accuracy),
                str(test_loss),
            ]
        )
        self.logger.log(level=lvl, msg=msg)
