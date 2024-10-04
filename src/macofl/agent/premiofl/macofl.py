from aioxmpp import JID
from spade.behaviour import CyclicBehaviour
from macofl.agent import AgentNodeBase


class MacoflAgent(AgentNodeBase):

    def __init__(
        self,
        jid: str,
        password: str,
        max_message_size: int,
        observers: list[JID] | None,
        neighbours: list[JID] | None,
        coordinator: JID | None = None,
        training_epochs: int = 1,
        consensus_iterations: int = 1,
        max_training_iterations: int = 100,
        web_address: str = "0.0.0.0",
        web_port: int = 10000,
        verify_security: bool = False,
    ):
        self.training_epochs = training_epochs
        self.consensus_iterations = consensus_iterations
        self.max_training_iterations = max_training_iterations
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
