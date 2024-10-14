from abc import ABCMeta, abstractmethod
from datetime import datetime, timezone
from queue import Queue
from typing import Optional, OrderedDict

from aioxmpp import JID
from spade.behaviour import CyclicBehaviour
from spade.message import Message
from spade.template import Template
from torch import Tensor

from ...datatypes.consensus import Consensus
from ...datatypes.consensus_transmission import ConsensusTransmission
from ...datatypes.models import ModelManager
from ...log.algorithm import AlgorithmLogManager
from ...log.message import MessageLogManager
from ...log.nn import NnInferenceLogManager, NnTrainLogManager
from ...similarity.similarity_manager import SimilarityManager
from ...similarity.similarity_vector import SimilarityVector
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
        extra_name = f"agent.{JID.fromstr(jid).localpart}"
        self.consensus = consensus
        self.model_manager = model_manager
        self.max_algorithm_iterations = max_algorithm_iterations  # None = inf
        self.similarity_manager: SimilarityManager = SimilarityManager(model_manager)
        self.algorithm_iterations: int = 0
        self.consensus_transmissions: Queue[ConsensusTransmission] = Queue()
        self.message_logger = MessageLogManager(extra_logger_name=extra_name)
        self.algorithm_logger = AlgorithmLogManager(extra_logger_name=extra_name)
        self.nn_train_logger = NnTrainLogManager(extra_logger_name=extra_name)
        self.nn_inference_logger = NnInferenceLogManager(extra_logger_name=extra_name)
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

    def select_neighbours(self) -> list[JID]:
        """
        Get the selected available neighbours to share the model layers, based on the implementation criteria.

        Raises:
            NotImplementedError: _select_neighbours must be overrided or it raises this error.

        Returns:
            list[JID]: The list of the selected available neighbours.
        """
        return self._select_neighbours(self.get_available_neighbours())

    @abstractmethod
    def _select_neighbours(self, neighbours: list[JID]) -> list[JID]:
        raise NotImplementedError

    @abstractmethod
    def assign_layers(
        self, neighbours: list[JID]
    ) -> dict[JID, OrderedDict[str, Tensor]]:
        """
        This function assigns which layers will be sent to each neighbour. In the paper it is coined as `S_L_N`.

        Args:
            neighbours (list[JID]): The neighbours that will receive the layers of the neural network model.

        Raises:
            NotImplementedError: This function must be overrided or it raises this error.

        Returns:
            dict[JID, OrderedDict[str, Tensor]]: The keys are the neighbour's `aioxmpp.JID`s and the values are the
            layer names with the `torch.Tensor` weights or biases.
        """
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
            self.consensus_transmissions.task_done()
        return consumed_consensus_transmissions

    async def send_similarity_vector(
        self,
        neighbour: JID,
        request_reply: bool,  # TODO
        vector: SimilarityVector,
        thread: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
        behaviour: Optional[CyclicBehaviour] = None,
    ) -> None:
        msg = vector.to_message()
        msg.sender = str(self.jid.bare())
        msg.to = str(neighbour.bare())
        msg.thread = thread
        msg.metadata = metadata
        await self.__send_message(
            message=msg, behaviour=behaviour, log_tag="-SIMILARITY"
        )

    async def send_local_layers(
        self,
        neighbour: JID,
        request_reply: bool,
        layers: OrderedDict[str, Tensor],
        thread: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
        behaviour: Optional[CyclicBehaviour] = None,
    ) -> None:
        ct = ConsensusTransmission(
            model=layers, sender=self.jid, request_reply=request_reply
        )
        msg = ct.to_message()
        msg.sender = str(self.jid.bare())
        msg.to = str(neighbour.bare())
        msg.thread = thread
        msg.metadata = metadata
        tag = "-REQREPLY" if request_reply else ""
        await self.__send_message(
            message=msg, behaviour=behaviour, log_tag=f"-LAYERS{tag}"
        )

    async def __send_message(
        self, message: Message, behaviour: CyclicBehaviour, log_tag: str = ""
    ) -> None:
        await self.send(message=message, behaviour=behaviour)
        self.message_logger.log(
            iteration_id=self.algorithm_iterations,
            sender=message.sender,
            to=message.to,
            msg_type=f"SEND{log_tag}",
            size=len(message.body),
            thread=message.thread,
        )

    def are_max_iterations_reached(self) -> bool:
        return (
            self.max_algorithm_iterations is not None
            and self.algorithm_iterations > self.max_algorithm_iterations
        )

    async def stop(self) -> None:
        await super().stop()
        self.logger.info(f"[{self.algorithm_iterations}] Agent stopped.")
