import random
from typing import OrderedDict

from aioxmpp import JID
from spade.template import Template
from torch import Tensor

from macofl.datatypes.consensus import Consensus
from macofl.datatypes.models import ModelManager

from ...behaviour.premiofl.fsm import AcolFsmBehaviour
from ...behaviour.premiofl.layer_receiver import LayerReceiverBehaviour
from .premiofl import PremioFlAgent


class AcolAgent(PremioFlAgent):

    def __init__(
        self,
        jid: str,
        password: str,
        max_message_size: int,
        consensus: Consensus,
        model_manager: ModelManager,
        observers: list[JID] | None = None,
        neighbours: list[JID] | None = None,
        coordinator: JID | None = None,
        max_algorithm_iterations: int | None = 100,
        web_address: str = "0.0.0.0",
        web_port: int = 10000,
        verify_security: bool = False,
    ):
        self.acol_fsm = AcolFsmBehaviour()
        self.acol_cyclic_receiver = LayerReceiverBehaviour()
        post_coordination_behaviours = [
            (self.acol_fsm, None),
            (
                self.acol_cyclic_receiver,
                Template(
                    metadata={"rf.conversation": "consensus_send_to_cyclic_receive"}
                ),
            ),
        ]
        super().__init__(
            jid,
            password,
            max_message_size,
            consensus,
            model_manager,
            post_coordination_behaviours,
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

    def assign_layers(
        self, neighbours: list[JID]
    ) -> dict[JID, OrderedDict[str, Tensor]]:
        return {n: self.model_manager.model.state_dict() for n in neighbours}
