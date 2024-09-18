from aioxmpp import JID
from typing import Coroutine, Any

from macofl.agent import AgentBase, AgentNodeBase
from macofl.behaviour import LaunchAgentsBehaviour, Wait


class LauncherAgent(AgentBase):

    def __init__(
        self,
        jid: str,
        password: str,
        max_message_size: int,
        agents_coordinator: JID,
        agents_observers: list[JID],
        agents_to_launch: list[JID],
        web_address: str = "0.0.0.0",
        web_port: int = 10000,
        verify_security: bool = False,
    ):
        self.agents: list[AgentNodeBase] = []
        self.agents_coordinator = agents_coordinator
        self.agents_observers = [] if agents_observers is None else agents_observers
        self.agents_to_launch = [] if agents_to_launch is None else agents_to_launch
        super().__init__(
            jid, password, max_message_size, web_address, web_port, verify_security
        )
        self.logger.debug("Initialized")

    async def setup(self) -> Coroutine[Any, Any, None]:
        self.setup_presence()
        self.presence.set_available()
        self.add_behaviour(LaunchAgentsBehaviour())
        self.add_behaviour(Wait())

    async def launch_agents(self) -> Coroutine[Any, Any, None]:
        for agent_jid in self.agents_to_launch:
            neighbour_jids = [j for j in self.agents_to_launch if j != agent_jid]
            agent = AgentNodeBase(
                jid=str(agent_jid.bare()),
                password="123",
                max_message_size=self.max_message_size,
                observers=self.agents_observers,
                neighbours=neighbour_jids,
                coordinator=self.agents_coordinator,
                verify_security=self.verify_security,
            )
            self.logger.debug(
                f"The neighbour JIDs for agent {agent_jid.bare()} are {[str(j.bare()) for j in neighbour_jids]}"
            )
            self.agents.append(agent)

        for agent in self.agents:
            await agent.start()
