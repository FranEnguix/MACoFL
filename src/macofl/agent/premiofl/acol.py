import random
from typing import Optional, OrderedDict

from aioxmpp import JID
from torch import Tensor

from macofl.datatypes.consensus_manager import ConsensusManager
from macofl.datatypes.models import ModelManager

from ...similarity.similarity_manager import SimilarityManager
from ...similarity.similarity_vector import SimilarityVector
from .premiofl import PremioFlAgent


class AcolAgent(PremioFlAgent):

    def __init__(
        self,
        jid: str,
        password: str,
        max_message_size: int,
        consensus_manager: ConsensusManager,
        model_manager: ModelManager,
        similarity_manager: SimilarityManager,
        observers: list[JID] | None = None,
        neighbours: list[JID] | None = None,
        coordinator: JID | None = None,
        max_algorithm_iterations: int | None = 100,
        web_address: str = "0.0.0.0",
        web_port: int = 10000,
        verify_security: bool = False,
    ):
        # self.acol_fsm = PremioFsmBehaviour()
        # self.acol_cyclic_receiver = LayerReceiverBehaviour()
        # post_coordination_behaviours = [
        #     (self.acol_fsm, None),
        #     (
        #         self.acol_cyclic_receiver,
        #         Template(metadata={"rf.conversation": "layers"}),
        #     ),
        # ]
        super().__init__(
            jid,
            password,
            max_message_size,
            consensus_manager,
            model_manager,
            similarity_manager,
            # post_coordination_behaviours,
            observers,
            neighbours,
            coordinator,
            max_algorithm_iterations,
            web_address,
            web_port,
            verify_security,
        )

    def _select_neighbours(self, neighbours: list[JID]) -> list[JID]:
        if not neighbours:
            return []
        return [random.choice(neighbours)]

    def _assign_layers(
        self,
        my_vector: Optional[SimilarityVector],
        neighbours_vectors: dict[JID, SimilarityVector],
        selected_neighbours: list[JID],
    ) -> dict[JID, OrderedDict[str, Tensor]]:
        return {n: self.model_manager.model.state_dict() for n in selected_neighbours}
