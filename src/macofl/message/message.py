import json
from datetime import datetime, timezone
from typing import Any, OrderedDict

from spade.message import Message
from torch import Tensor


class MessageFl(Message):

    def __init__(self, message: Message):
        to = message.to
        sender = message.sender
        body = message.body
        thread = message.thread
        metadata = message.metadata
        super().__init__(to, sender, body, thread, metadata)
        self.id: str
        self.timestamp_z: datetime
        self.weights: OrderedDict[str, Tensor]
        self.load(message.body)

    def load(self, json_text: str) -> None:
        content: dict[str, Any] = json.loads(json_text)
        self.id = content["id"]
        self.timestamp_z = datetime.strptime(
            content["timestamp"], "%Y-%m-%d %H:%M:%S.%f"
        ).replace(tzinfo=timezone.utc)
        base64_weights: str = content["weights"]

    def to_json(self) -> str:
        content: dict[str, Any] = {}
        return json.dumps(content)
