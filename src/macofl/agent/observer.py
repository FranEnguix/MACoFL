from aioxmpp import JID
from typing import Coroutine, Any
from spade.template import Template

from macofl.agent import AgentBase
from macofl.behaviour.observer import ObserverBehaviour


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
        self.observation_theme_behaviours = {
            "message": None,
            "accuracy": None,
            "loss": None,
            "iteration": None,
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
            behaviour = ObserverBehaviour(f"rf.{theme}.{self.name}")
            self.observation_theme_behaviours[theme] = behaviour
            self.add_behaviour(behaviour, template)
