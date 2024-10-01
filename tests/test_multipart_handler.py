import math
import random

from spade.message import Message

from macofl.message import MultipartHandler


def test_divide_and_rebuild_message(
    content_multiplier: int = 30, max_content_size: int = 10
):
    mh_sender = MultipartHandler()
    mh_dest = MultipartHandler()

    original_content = ""
    for i in range(content_multiplier):
        original_content += f"{i}#/|sdf|/#multipart|"

    to = "dest"
    sender = "sender"
    thread = "th"
    meta = {"uno": 1, "dos": "2"}
    max_size_with_header = max_content_size + mh_sender.metadata_header_size

    msg = Message(to=to, sender=sender, body=original_content)
    msg.thread = thread
    msg.metadata = meta

    msgs = mh_sender.generate_multipart_messages(
        content=original_content, max_size=max_size_with_header, message_base=msg
    )
    msgs = [] if msgs is None else msgs
    random.shuffle(msgs)

    assert len(msgs) == math.ceil(
        len(original_content) / (max_size_with_header - mh_sender.metadata_header_size)
    )

    result: Message = None
    for m in msgs:
        result = mh_dest.rebuild_multipart(m)

    assert result is not None and result.body == original_content
