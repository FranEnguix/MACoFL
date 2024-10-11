from aioxmpp import JID

from macofl.agent import AgentBase
from macofl.agent.premiofl import AcolAgent
from macofl.behaviour.launcher import LaunchAgentsBehaviour, Wait

from ..datatypes.consensus import Consensus
from ..nn.model_factory import ModelManagerFactory
from .base import AgentNodeBase


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
        self.agents: list[AcolAgent] = []
        self.agents_coordinator = agents_coordinator
        self.agents_observers = [] if agents_observers is None else agents_observers
        self.agents_to_launch = [] if agents_to_launch is None else agents_to_launch
        super().__init__(
            jid, password, max_message_size, web_address, web_port, verify_security
        )
        self.logger.debug(
            f"Agents to launch: {[j.bare() for j in self.agents_to_launch]}"
        )

    async def setup(self) -> None:
        self.setup_presence_handlers()
        self.presence.set_available()
        self.add_behaviour(LaunchAgentsBehaviour())
        self.add_behaviour(Wait())

    async def launch_agents(self) -> None:
        self.logger.debug(
            f"Initializating launch of {[str(j.bare()) for j in self.agents_to_launch]}"
        )
        for agent_jid in self.agents_to_launch:
            neighbour_jids = [j for j in self.agents_to_launch if j != agent_jid]
            # agent = AgentNodeBase(
            #     jid=str(agent_jid.bare()),
            #     password="123",
            #     max_message_size=self.max_message_size,
            #     observers=self.agents_observers,
            #     neighbours=neighbour_jids,
            #     coordinator=self.agents_coordinator,
            #     verify_security=self.verify_security,
            # )
            consensus = Consensus(
                max_order=len(neighbour_jids), max_seconds_to_accept_pre_consensus=600
            )
            agent_index = int(str(agent_jid.localpart)[1])
            model_manager = ModelManagerFactory.get_cifar10(
                iid=False, client_index=agent_index, seed=42
            )
            agent = AcolAgent(
                jid=str(agent_jid.bare()),
                password="123",
                max_message_size=self.max_message_size,
                consensus=consensus,
                model_manager=model_manager,
                observers=self.agents_observers,
                neighbours=neighbour_jids,
                coordinator=self.agents_coordinator,
                max_algorithm_iterations=5,
            )
            self.logger.debug(
                f"The neighbour JIDs for agent {agent_jid.bare()} are {[str(j.bare()) for j in neighbour_jids]}"
            )
            self.agents.append(agent)

        for agent in self.agents:
            await agent.start()
