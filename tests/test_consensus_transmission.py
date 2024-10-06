import json
import time
import unittest
from datetime import datetime, timezone
from typing import Optional

import torch
from aioxmpp import JID
from spade.message import Message
from torch import nn

from macofl.datatypes import ModelManager
from macofl.datatypes.consensus_transmission import ConsensusTransmission

from .test_nn_model import build_neural_network


class TestConsensusTransmission(unittest.TestCase):

    def test_initialization(self):
        # Build a simple neural network model and get its state_dict
        model = nn.Linear(10, 5)
        model_state = model.state_dict()

        sender = JID.fromstr("sender@localhost")
        now = datetime.now(tz=timezone.utc)

        consensus_transmission = ConsensusTransmission(
            model=model_state, sender=sender, sent_time_z=now
        )

        self.assertEqual(consensus_transmission.model, model_state)
        self.assertEqual(consensus_transmission.sender, sender)
        self.assertEqual(consensus_transmission.sent_time_z, now)
        self.assertIsNone(consensus_transmission.received_time_z)
        self.assertIsNone(consensus_transmission.processed_start_time_z)
        self.assertIsNone(consensus_transmission.processed_end_time_z)

    def test_to_message(self):
        model = nn.Linear(10, 5)
        model_state = model.state_dict()

        sender = JID.fromstr("sender@localhost")
        now = datetime.now(tz=timezone.utc)

        consensus_transmission = ConsensusTransmission(
            model=model_state, sender=sender, sent_time_z=now
        )

        message = consensus_transmission.to_message()

        # Check that the message is an instance of Message
        self.assertIsInstance(message, Message)

        # Parse the message body
        content = json.loads(message.body)

        # Check that 'model' and 'sent_time_z' are in the content
        self.assertIn("model", content)
        self.assertIn("sent_time_z", content)

        # Verify that 'sent_time_z' is correctly formatted
        sent_time_str = content["sent_time_z"]
        try:
            parsed_time = datetime.strptime(sent_time_str, "%Y-%m-%dT%H:%M:%S.%fZ")
            self.assertEqual(parsed_time.tzinfo, None)  # No timezone info in the string
        except ValueError:
            self.fail("sent_time_z is not properly formatted")

    def test_from_message(self):
        model = nn.Linear(10, 5)
        model_state = model.state_dict()
        sender = JID.fromstr("sender@localhost")
        now = datetime.now(tz=timezone.utc)

        # Create a message
        consensus_transmission = ConsensusTransmission(
            model=model_state, sender=sender, sent_time_z=now
        )
        message = consensus_transmission.to_message()
        message.sender = str(sender.bare())  # Simulate sender

        # Deserialize from message
        received_transmission = ConsensusTransmission.from_message(message)

        # Check that the fields are correctly set
        for key in model_state.keys():
            assert torch.allclose(
                received_transmission.model[key], model_state.get(key)
            )
        self.assertEqual(received_transmission.sender, sender)
        self.assertEqual(
            received_transmission.sent_time_z, consensus_transmission.sent_time_z
        )
        self.assertIsNotNone(received_transmission.received_time_z)
        self.assertIsNone(received_transmission.processed_start_time_z)
        self.assertIsNone(received_transmission.processed_end_time_z)

        # Check that received_time_z is set and is a datetime with timezone
        self.assertIsInstance(received_transmission.received_time_z, datetime)
        self.assertIsNotNone(received_transmission.received_time_z.tzinfo)

    def test_round_trip(self):
        model = nn.Linear(10, 5)
        model_state = model.state_dict()
        sender = JID.fromstr("sender@localhost")
        now = datetime.now(tz=timezone.utc)

        consensus_transmission = ConsensusTransmission(
            model=model_state, sender=sender, sent_time_z=now
        )

        time.sleep(0.2)  # Pause to check if sent_time_z is not override in to_message()

        message = consensus_transmission.to_message()
        message.sender = str(sender.bare())  # Simulate sender

        time.sleep(0.2)  # Simulate send time

        received_transmission = ConsensusTransmission.from_message(message)

        # Since received_time_z will be different, we can compare the rest of the fields
        for key in model_state.keys():
            assert torch.allclose(
                received_transmission.model[key], consensus_transmission.model[key]
            )
        self.assertEqual(received_transmission.sender, consensus_transmission.sender)
        self.assertEqual(
            received_transmission.sent_time_z, consensus_transmission.sent_time_z
        )

    def test_datetime_format(self):
        now = datetime.now(tz=timezone.utc)
        formatted_time = now.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        time.sleep(0.2)
        parsed_time = datetime.strptime(
            formatted_time, "%Y-%m-%dT%H:%M:%S.%fZ"
        ).replace(tzinfo=timezone.utc)

        self.assertEqual(now, parsed_time)

    def test_str_method(self):
        model = nn.Linear(10, 5)
        model_state = model.state_dict()
        sender = JID.fromstr("sender@localhost")
        now = datetime.now(tz=timezone.utc)

        consensus_transmission = ConsensusTransmission(
            model=model_state,
            sender=sender,
            sent_time_z=now,
            received_time_z=now,
            processed_start_time_z=now,
            processed_end_time_z=now,
        )

        result_str = str(consensus_transmission)
        content = json.loads(result_str)

        # Check that all fields are present
        self.assertIn("model", content)
        self.assertIn("sender", content)
        self.assertIn("sent_time_z", content)
        self.assertIn("received_time_z", content)
        self.assertIn("processed_start_time_z", content)
        self.assertIn("processed_end_time_z", content)

        # Verify datetime fields
        for field in [
            "sent_time_z",
            "received_time_z",
            "processed_start_time_z",
            "processed_end_time_z",
        ]:
            time_str = content[field]
            parsed_time = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%fZ")
            self.assertEqual(parsed_time.tzinfo, None)  # No timezone info in the string

    def test_timezone_handling(self):
        # Timezone-aware datetime
        aware_now = datetime.now(tz=timezone.utc)
        # Timezone-naive datetime
        naive_now = datetime.now()

        # Create instances with both
        sender = JID.fromstr("sender@localhost")
        model = nn.Linear(10, 5).state_dict()

        # With aware datetime
        consensus_transmission_aware = ConsensusTransmission(
            model=model, sender=sender, sent_time_z=aware_now
        )

        message_aware = consensus_transmission_aware.to_message()
        self.assertIn("sent_time_z", json.loads(message_aware.body))

        # With naive datetime (should handle or raise an error)
        with self.assertRaises(ValueError):
            # This should fail because the datetime is naive
            ConsensusTransmission(model=model, sender=sender, sent_time_z=naive_now)

    def test_build_consensus_transmission(
        self, sender: Optional[JID] = None, sent_time_z: Optional[datetime] = None
    ) -> ConsensusTransmission:
        model: ModelManager = build_neural_network()
        sender = JID.fromstr("sender@localhost") if sender is None else sender
        now = datetime.now(tz=timezone.utc) if sent_time_z is None else sent_time_z
        consensus_transmission = ConsensusTransmission(
            model=model.initial_state, sender=sender, sent_time_z=now
        )
        return consensus_transmission

    def test_consensus_transmission_timezone_handling(self) -> None:
        # Timezone-aware datetime
        aware_now = datetime.now(tz=timezone.utc)
        # Timezone-naive datetime
        naive_now = datetime.now()

        # Create instances with both
        sender = JID.fromstr("sender@localhost")
        model = nn.Linear(10, 5).state_dict()

        # With aware datetime
        consensus_transmission_aware = ConsensusTransmission(
            model=model, sender=sender, sent_time_z=aware_now
        )

        message_aware = consensus_transmission_aware.to_message()
        self.assertIn("sent_time_z", json.loads(message_aware.body))

        # With naive datetime (should handle or raise an error)
        with self.assertRaises(ValueError):
            # This should fail because the datetime is naive
            ConsensusTransmission(model=model, sender=sender, sent_time_z=naive_now)
