import traceback
from abc import ABCMeta, abstractmethod
from datetime import datetime, timezone
from queue import Queue
from typing import Optional, OrderedDict

from aioxmpp import JID
from spade.behaviour import CyclicBehaviour
from spade.template import Template
from torch import Tensor

from ...datatypes.consensus import Consensus
from ...datatypes.consensus_transmission import ConsensusTransmission
from ...datatypes.models import ModelManager
from ..base import AgentNodeBase


class PremioFlAgent(AgentNodeBase, metaclass=ABCMeta):

    def __init__(
        self,
        jid: str,
        password: str,
        max_message_size: int,
        consensus: Consensus,
        model_manager: ModelManager,
        post_coordination_behaviours: Optional[list[tuple[CyclicBehaviour, Template]]],
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
        self.algorithm_iterations: int = 0
        self.consensus_transmissions: Queue[ConsensusTransmission] = Queue()
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

    @abstractmethod
    def select_neighbours(self) -> list[JID]:
        raise NotImplementedError

    @abstractmethod
    def select_layers(self) -> OrderedDict[str, Tensor]:
        raise NotImplementedError

    def put_model_to_consensus_queue(self, model: OrderedDict[str, Tensor]) -> None:
        self.consensus.models_to_consensuate.put(model)

    def apply_all_consensus(
        self, other_model: Optional[OrderedDict[str, Tensor]] = None
    ) -> None:
        if other_model is not None:
            self.consensus.models_to_consensuate.put(other_model)
        weights_and_biases = self.model_manager.model.state_dict()
        consensuated_model = self.consensus.apply_all_consensus(
            model=weights_and_biases
        )
        self.model_manager.replace_weights_and_biases(
            new_weights_and_biases=consensuated_model
        )

    def apply_consensus(self, other_model: OrderedDict[str, Tensor]) -> None:
        weights_and_biases = self.model_manager.model.state_dict()
        consensuated_model = Consensus.apply_consensus(
            weights_and_biases_a=weights_and_biases,
            weights_and_biases_b=other_model,
            max_order=self.consensus.max_order,
        )
        self.model_manager.replace_weights_and_biases(
            new_weights_and_biases=consensuated_model
        )

    def put_to_consensus_transmission_queue(
        self, consensus_transmission: ConsensusTransmission
    ) -> None:
        self.consensus_transmissions.put(consensus_transmission)

    async def apply_all_consensus_transmission(
        self,
        consensus_transmission: Optional[ConsensusTransmission] = None,
        send_model_during_consensus: bool = False,
        metadata: Optional[dict[str, str]] = None,
        behaviour: Optional[CyclicBehaviour] = None,
    ) -> list[ConsensusTransmission]:
        consumed_consensus_transmissions: list[JID] = []
        if consensus_transmission is not None:
            self.consensus_transmissions.put(consensus_transmission)
        while self.consensus_transmissions.qsize() > 0:
            ct = self.consensus_transmissions.get()
            ct.processed_start_time_z = datetime.now(tz=timezone.utc)
            self.apply_consensus(ct.model)
            ct.processed_end_time_z = datetime.now(tz=timezone.utc)
            consumed_consensus_transmissions.append(ct)
            if send_model_during_consensus:
                await self.send_local_weights(
                    neighbour=ct.sender, metadata=metadata, behaviour=behaviour
                )
            self.consensus_transmissions.task_done()
        return consumed_consensus_transmissions

    async def send_local_weights(
        self,
        neighbour: JID,
        metadata: Optional[dict[str, str]] = None,
        behaviour: Optional[CyclicBehaviour] = None,
    ) -> None:
        model = self.select_layers()
        ct = ConsensusTransmission(
            model=model,
            sender=self.jid,
        )
        msg = ct.to_message()
        msg.to = str(neighbour.bare())
        if metadata is not None:
            msg.metadata = metadata
        await self.send(message=msg, behaviour=behaviour)
