from typing import OrderedDict

from aioxmpp import JID
from spade.behaviour import CyclicBehaviour
from torch import Tensor

from ...datatypes.consensus import Consensus
from ..base import AgentNodeBase


class PremioFlAgent(AgentNodeBase):

    def __init__(
        self,
        jid: str,
        password: str,
        max_message_size: int,
        consensus: Consensus,
        observers: list[JID] | None = None,
        neighbours: list[JID] | None = None,
        coordinator: JID | None = None,
        training_epochs: int = 1,
        max_algorithm_iterations: int = 100,
        web_address: str = "0.0.0.0",
        web_port: int = 10000,
        verify_security: bool = False,
    ):
        self.consensus = consensus
        self.training_epochs = training_epochs
        self.max_training_iterations = max_algorithm_iterations
        post_coordination_behaviours: list[CyclicBehaviour] = []
        self.weights = None
        self.max_order: int = 0
        super().__init__(
            jid,
            password,
            max_message_size,
            observers,
            neighbours,
            coordinator,
            post_coordination_behaviours,
            web_address,
            web_port,
            verify_security,
        )

    def is_training(self) -> bool:
        return True  # TODO

    def store_consensus_weights(self, weights: Tensor) -> None:
        self.consensus.weights.put(weights)

    def apply_consensus(self) -> None:
        while not self.consensus.weights.empty():
            weights = self.consensus.weights.get()

    async def send_local_weights(self) -> None:
        pass  # TODO
