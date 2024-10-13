import copy
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, OrderedDict

from aioxmpp import JID
from spade.message import Message


@dataclass
class SimilarityVector:
    vector: OrderedDict[
        str, float
    ]  # str is the name of the layer and float is the similarity coefficient
    owner: Optional[JID] = None
    sent_time_z: Optional[datetime] = None
    received_time_z: Optional[datetime] = None

    def to_message(self, message: Optional[Message] = None) -> Message:
        msg = Message() if message is None else copy.deepcopy(message)
        content: dict[str, Any] = {}
        content["vector"] = json.dumps(self.vector)
        sent_time_z = (
            datetime.now(tz=timezone.utc)
            if self.sent_time_z is None
            else self.sent_time_z
        )
        content["sent_time_z"] = sent_time_z.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        msg.body = json.dumps(content)
        return msg

    @staticmethod
    def from_message(message: Message) -> "SimilarityVector":
        content: dict[str, Any] = json.loads(message.body)
        vector: OrderedDict[str, float] = content["vector"]
        sent_time_z: datetime = datetime.strptime(
            content["sent_time_z"], "%Y-%m-%dT%H:%M:%S.%fZ"
        ).replace(tzinfo=timezone.utc)
        received_time_z: datetime = datetime.now(tz=timezone.utc)
        if "received_time_z" in content:
            received_time_z = datetime.strptime(
                content["received_time_z"], "%Y-%m-%dT%H:%M:%S.%fZ"
            ).replace(tzinfo=timezone.utc)
        return SimilarityVector(
            vector=vector,
            owner=message.sender,
            sent_time_z=sent_time_z,
            received_time_z=received_time_z,
        )
