import math
import random

import torch
from spade.message import Message

from macofl.datatypes import ModelManager
from macofl.message import MultipartHandler

from .test_nn_model import build_neural_network


def test_divide_and_rebuild_message(
    content_multiplier: int = 30, max_content_size: int = 10
) -> None:
    mh_sender = MultipartHandler()
    mh_dest = MultipartHandler()

    original_content = "".join(
        [f"{i}#/|sdf|/#multipart|" for i in range(content_multiplier)]
    )

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


def test_model_message_transmission() -> None:
    mh_sender = MultipartHandler()
    mh_dest = MultipartHandler()

    max_content_size = 250_000

    model = build_neural_network()
    original_content = ModelManager.export_weights_and_biases(model.model.state_dict())

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

    print(f"Messages of {max_content_size}: {len(msgs)}")
    assert len(msgs) == math.ceil(
        len(original_content) / (max_size_with_header - mh_sender.metadata_header_size)
    )

    result: Message = None
    for m in msgs:
        result = mh_dest.rebuild_multipart(m)

    assert result is not None and result.body == original_content
    model_reconstruct = ModelManager.import_weights_and_biases(result.body)
    for key in model.initial_state:
        assert key in model_reconstruct, f"Key '{key}' not in reconstruct."
        assert torch.allclose(
            model.model.state_dict()[key], model_reconstruct[key]
        ), f"Reconstructed '{key}' tensor does not match the model"
        assert torch.allclose(
            model.initial_state[key], model_reconstruct[key]
        ), f"Reconstructed '{key}' tensor does not match the initial model"
