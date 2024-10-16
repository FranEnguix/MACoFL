from aioxmpp import JID

from ..behaviour.launcher import LaunchAgentsBehaviour, Wait
from ..datatypes.consensus_manager import ConsensusManager
from ..datatypes.data import NonIidDirichletDatasetSettings
from ..nn.model_factory import ModelManagerFactory
from ..similarity.function import EuclideanDistanceFunction
from ..similarity.similarity_manager import SimilarityManager
from .base import AgentBase
from .premiofl.pmacofl_min import PmacoflMinAgent


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
        self.agents: list[PmacoflMinAgent] = []
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
        max_order = len(self.agents_to_launch) - 1
        for agent_jid in self.agents_to_launch:
            neighbour_jids = [j for j in self.agents_to_launch if j != agent_jid]

            agent_index = int(str(agent_jid.localpart)[1])
            dataset_settings = NonIidDirichletDatasetSettings(
                seed=13,
                num_clients=len(self.agents_to_launch),
                client_index=agent_index,
            )
            model_manager = ModelManagerFactory.get_cifar10_cnn5(
                settings=dataset_settings
            )
            consensus = ConsensusManager(
                model_manager=model_manager,
                max_order=max_order,
                max_seconds_to_accept_consensus=24 * 60 * 60,
                consensus_iterations=10,
            )
            similarity_manager = SimilarityManager(
                model_manager=model_manager,
                function=EuclideanDistanceFunction(),
                wait_for_responses_timeout=5 * 60,
            )
            agent = PmacoflMinAgent(
                jid=str(agent_jid.bare()),
                password="123",
                max_message_size=self.max_message_size,
                consensus_manager=consensus,
                model_manager=model_manager,
                similarity_manager=similarity_manager,
                observers=self.agents_observers,
                neighbours=neighbour_jids,
                coordinator=self.agents_coordinator,
                max_rounds=70,
            )
            self.logger.debug(
                f"The neighbour JIDs for agent {agent_jid.bare()} are {[str(j.bare()) for j in neighbour_jids]}"
            )
            self.agents.append(agent)

        for agent in self.agents:
            await agent.start()
