from typing import Optional, OrderedDict

from aioxmpp import JID
from spade.behaviour import CyclicBehaviour
from torch import Tensor

from ...datatypes.consensus import Consensus
from ...datatypes.models import ModelManager
from ..base import AgentNodeBase


class PremioFlAgent(AgentNodeBase):

    def __init__(
        self,
        jid: str,
        password: str,
        max_message_size: int,
        consensus: Consensus,
        model_manager: ModelManager,
        post_coordination_behaviours: list[CyclicBehaviour],
        observers: list[JID] | None = None,
        neighbours: list[JID] | None = None,
        coordinator: JID | None = None,
        max_algorithm_iterations: Optional[int] = 100,
        web_address: str = "0.0.0.0",
        web_port: int = 10000,
        verify_security: bool = False,
    ):
        self.consensus = consensus
        self.model_manager = model_manager
        self.max_algorithm_iterations = max_algorithm_iterations  # None = inf
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
        self.model: ModelManager

    def put_model_to_consensus_queue(self, model: OrderedDict[str, Tensor]) -> None:
        self.consensus.models_to_consensuate.put(model)

    def apply_consensus(
        self, other_model: Optional[OrderedDict[str, Tensor]] = None
    ) -> None:
        if other_model is not None:
            self.consensus.models_to_consensuate.put(other_model)
        weights_and_biases = self.model.model.state_dict()
        consensuated_model = self.consensus.apply_all_consensus(
            model=weights_and_biases
        )
        self.model.replace_weights_and_biases(new_weights_and_biases=consensuated_model)

    async def send_local_weights(self, neighbour: JID) -> None:
        pass  # TODO
