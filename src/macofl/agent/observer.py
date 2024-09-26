from aioxmpp import JID
from typing import Coroutine, Any
from spade.template import Template

from macofl.agent import AgentBase
from macofl.behaviour.coordination import PresenceCoordinatorFSM


class ObserverAgent(AgentBase):

    def __init__(
        self,
        jid: str,
        password: str,
        max_message_size: int,
        web_address: str = "0.0.0.0",
        web_port: int = 10000,
        verify_security: bool = False,
    ):
        self.agents_observed: list[JID] = []
        self.coordination_fsm: PresenceCoordinatorFSM = None
        self.observation_theme_behaviours = {
            "message_sent": None,
            "message_received": None,
            "iteration_time": None,
            "accuracy": None,
            "loss": None,
        }
        super().__init__(
            jid,
            password,
            max_message_size,
            web_address,
            web_port,
            verify_security,
        )

    async def setup(self) -> Coroutine[Any, Any, None]:
        await super().setup()
        for theme in self.observation_theme_behaviours.keys():
            template = Template(metadata={"rf.observe": theme})
            self.observation_theme_behaviours()
        self.coordination_fsm = PresenceCoordinatorFSM(self.coordinated_agents)
        self.add_behaviour(self.coordination_fsm, template)
