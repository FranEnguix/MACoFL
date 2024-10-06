import copy
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, OrderedDict

from aioxmpp import JID
from spade.message import Message
from torch import Tensor

from ..datatypes.models import ModelManager


@dataclass
class ConsensusTransmission:
    """
    Dataclass to store consensus information during model transmission and processing.
    """

    model: OrderedDict[str, Tensor]
    sender: JID
    sent_time_z: Optional[datetime] = None
    received_time_z: Optional[datetime] = None
    processed_start_time_z: Optional[datetime] = None
    processed_end_time_z: Optional[datetime] = None

    def __post_init__(self) -> None:
        self.__check_utc(self.sent_time_z)
        self.__check_utc(self.received_time_z)
        self.__check_utc(self.processed_start_time_z)
        self.__check_utc(self.processed_end_time_z)

    def to_message(self, message: Optional[Message] = None) -> Message:
        msg = Message() if message is None else copy.deepcopy(message)
        content: dict[str, Any] = {}
        base64_model = ModelManager.export_weights_and_biases(self.model)
        content["model"] = base64_model
        sent_time_z = (
            datetime.now(tz=timezone.utc)
            if self.sent_time_z is None
            else self.sent_time_z
        )
        content["sent_time_z"] = sent_time_z.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        msg.body = json.dumps(content)
        return msg

    @staticmethod
    def from_message(message: Message) -> "ConsensusTransmission":
        content: dict[str, Any] = json.loads(message.body)
        base64_model: str = content["model"]
        model = ModelManager.import_weights_and_biases(base64_model)
        sent_time_z: datetime = datetime.strptime(
            content["sent_time_z"], "%Y-%m-%dT%H:%M:%S.%fZ"
        ).replace(tzinfo=timezone.utc)
        received_time_z: datetime = datetime.now(tz=timezone.utc)
        if "received_time_z" in content:
            received_time_z = datetime.strptime(
                content["received_time_z"], "%Y-%m-%dT%H:%M:%S.%fZ"
            ).replace(tzinfo=timezone.utc)
        processed_start_time_z: Optional[datetime] = None
        if "processed_start_time_z" in content:
            processed_start_time_z = datetime.strptime(
                content["processed_start_time_z"], "%Y-%m-%dT%H:%M:%S.%fZ"
            ).replace(tzinfo=timezone.utc)
        processed_end_time_z: Optional[datetime] = None
        if "processed_end_time_z" in content:
            processed_end_time_z = datetime.strptime(
                content["processed_end_time_z"], "%Y-%m-%dT%H:%M:%S.%fZ"
            ).replace(tzinfo=timezone.utc)
        return ConsensusTransmission(
            model=model,
            sender=message.sender,
            sent_time_z=sent_time_z,
            received_time_z=received_time_z,
            processed_start_time_z=processed_start_time_z,
            processed_end_time_z=processed_end_time_z,
        )

    def __str__(self) -> str:
        content: dict[str, Any] = {}
        base64_model = ModelManager.export_weights_and_biases(self.model)
        content["model"] = base64_model
        content["sender"] = str(self.sender)
        if self.sent_time_z is not None:
            content["sent_time_z"] = self.sent_time_z.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        if self.received_time_z is not None:
            content["received_time_z"] = self.received_time_z.strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ"
            )
        if self.processed_start_time_z is not None:
            content["processed_start_time_z"] = self.processed_start_time_z.strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ"
            )
        if self.processed_end_time_z is not None:
            content["processed_end_time_z"] = self.processed_end_time_z.strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ"
            )
        return json.dumps(content)

    def __check_utc(self, dt: Optional[datetime]) -> None:
        if dt is not None:
            if dt.tzinfo is None or dt.tzinfo != timezone.utc:
                raise ValueError(
                    "All ConsensusTransmission datetimes must be timezone-aware (UTC) (Z)."
                )
