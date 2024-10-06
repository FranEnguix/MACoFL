from datetime import datetime, timezone
from queue import Queue
from typing import Callable, Optional, OrderedDict

from aioxmpp import JID
from spade.behaviour import CyclicBehaviour
from torch import Tensor

from ...datatypes.consensus import Consensus
from ...datatypes.consensus_transmission import ConsensusTransmission
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
        select_neighbours: Callable[[None], list[JID]],
        select_layers: Callable[[None], OrderedDict[str, Tensor]],
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
        self.consensus_transmission: Queue[ConsensusTransmission] = Queue()
        self.select_neighbours = select_neighbours
        self.select_layers = select_layers
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

    def apply_all_consensus(
        self, other_model: Optional[OrderedDict[str, Tensor]] = None
    ) -> None:
        if other_model is not None:
            self.consensus.models_to_consensuate.put(other_model)
        weights_and_biases = self.model.model.state_dict()
        consensuated_model = self.consensus.apply_all_consensus(
            model=weights_and_biases
        )
        self.model.replace_weights_and_biases(new_weights_and_biases=consensuated_model)

    def apply_consensus(self, other_model: OrderedDict[str, Tensor]) -> None:
        weights_and_biases = self.model.model.state_dict()
        consensuated_model = Consensus.apply_consensus(
            weights_and_biases_a=weights_and_biases,
            weights_and_biases_b=other_model,
            epsilon=self.consensus.epsilon,
        )
        self.model.replace_weights_and_biases(new_weights_and_biases=consensuated_model)

    def put_to_consensus_transmission_queue(
        self, consensus_transmission: ConsensusTransmission
    ) -> None:
        self.consensus_transmission.put(consensus_transmission)

    async def apply_all_consensus_transmission(
        self, consensus_transmission: Optional[ConsensusTransmission] = None
    ) -> None:
        if consensus_transmission is not None:
            self.consensus_transmission.put(consensus_transmission)
        while self.consensus_transmission.qsize() > 0:
            ct = self.consensus_transmission.get()
            ct.processed_start_time_z = datetime.now(tz=timezone.utc)
            self.apply_consensus(ct.model)
            ct.processed_end_time_z = datetime.now(tz=timezone.utc)
            await self.send_local_weights(ct.sender)
            self.consensus_transmission.task_done()

    async def send_local_weights(
        self,
        neighbour: JID,
        metadata: Optional[dict[str, str]] = None,
        behaviour: Optional[CyclicBehaviour] = None,
    ) -> None:
        ct = ConsensusTransmission(
            model=self.model_manager.model.state_dict(),
            sender=self.jid,
        )
        msg = ct.to_message()
        msg.to = str(neighbour.bare())
        if metadata is not None:
            msg.metadata = metadata
        await self.send(message=msg, behaviour=behaviour)
