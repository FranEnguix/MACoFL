import asyncio
from typing import Coroutine, Any, TYPE_CHECKING

from aioxmpp import JID
from spade.behaviour import FSMBehaviour, State
from spade.message import Message

if TYPE_CHECKING:
    from macofl.agent.coordinator import CoordinatorAgent
    from macofl.agent import AgentNodeBase


class AvailableNodeState(State):

    def __init__(self, coordinator: JID):
        self.coordinator = coordinator
        self.loops: int = 0
        super().__init__()

    async def on_start(self) -> None:
        if self.loops < 1:
            agent: AgentNodeBase = self.agent
            agent.presence.set_available()
            agent.logger.debug(f"Available.")
            coordinator = str(self.coordinator.bare())
            message = Message(to=coordinator, sender=str(agent.jid.bare()))
            message.body = "ready to subscribe"
            message.set_metadata("presence", "sync")
            await self.send(message)
            agent.logger.debug(f"Ready message sent to {coordinator}.")
        self.loops += 1

    async def run(self) -> None:
        agent: AgentNodeBase = self.agent
        self.set_next_state("available")
        msg = await self.receive(timeout=1)
        if msg and msg.body == "start to subscribe":
            jid = msg.sender.bare()
            agent.logger.debug(f"Start to subscribe message received from {jid}.")
            self.set_next_state("subscription")


class SubscriptionNodeState(State):

    def __init__(self, coordinator: JID):
        self.coordinator = coordinator
        self.loops: int = 0
        self.ready_to_start = False
        super().__init__()

    async def on_start(self) -> None:
        if self.loops < 1:
            agent: AgentNodeBase = self.agent
            agent.logger.debug(f"Subscribing to neighbours...")
            agent.subscribe_to_neighbours()
        self.loops += 1

    async def run(self) -> None:
        agent: AgentNodeBase = self.agent
        if not self.ready_to_start and agent.is_presence_completed():
            self.ready_to_start = True
            coordinator = str(self.coordinator.bare())
            message = Message(to=coordinator, sender=str(agent.jid.bare()))
            message.body = "ready to start"
            message.set_metadata("presence", "sync")
            await self.send(message)
            agent.logger.debug(f"'ready to start' message sent to {coordinator}")
        elif not agent.is_presence_completed():
            # agent.logger.debug(f"Neighbour's subscription status is {subscription_status}")
            contacts: dict[JID, dict] = agent.presence.get_contacts()
            subscription_status = [
                (str(jid.bare()), data["subscription"])
                for jid, data in contacts.items()
            ]
            agent.logger.debug(
                f"Neighbour's subscription status is {subscription_status}"
            )
            for jid, status in agent.get_non_subscribe_both_neighbours().items():
                agent.presence.subscribe(jid)
                agent.logger.debug(
                    f"Sent subscription request to {jid} because status is '{status}'."
                )
            await asyncio.sleep(2)

        msg = await self.receive(timeout=1)
        if msg and msg.body == "start the algorithm":
            jid = msg.sender.bare()
            agent.logger.debug(f"'start the algorithm' message received from {jid}.")
            for behaviour in agent.post_coordination_behaviours:
                agent.add_behaviour(behaviour)
                agent.logger.debug(f"Behaviour {type(behaviour)} added.")
            self.kill()
        else:
            self.set_next_state("subscription")
        # self.set_next_state("wait")


class PresenceNodeFSM(FSMBehaviour):

    def __init__(self, coordinator: JID):
        self.coordinator = coordinator
        super().__init__()

    def setup(self) -> None:
        self.add_state(
            name="available", state=AvailableNodeState(self.coordinator), initial=True
        )
        self.add_state(
            name="subscription", state=SubscriptionNodeState(self.coordinator)
        )
        self.add_transition(source="available", dest="available")
        self.add_transition(source="available", dest="subscription")
        self.add_transition(source="subscription", dest="subscription")

    async def on_end(self) -> Coroutine[Any, Any, None]:
        agent: AgentNodeBase = self.agent
        agent.logger.debug(f"PresenceSetupFSM behaviour finished.")


# --------------------------------------------- #
# --------------------------------------------- #
# --------------------------------------------- #


class AvailableCoordinatorState(State):

    def __init__(self, coordinated_agents: list[JID]):
        self.loop: int = 0
        self.ready_agents: dict[str, bool] = {
            str(jid.bare()): False for jid in coordinated_agents
        }
        super().__init__()

    async def on_start(self) -> None:
        if self.loop < 1:
            agent: CoordinatorAgent = self.agent
            agent.logger.debug(f"AvailableCoordinatorState: {self.ready_agents}.")
        self.loop += 1

    async def run(self) -> None:
        agent: CoordinatorAgent = self.agent
        self.set_next_state("available")
        if not self._are_all_agents_ready():
            msg = await self.receive(timeout=3)
            if msg and msg.body == "ready to subscribe":
                jid = msg.sender.bare()
                self.ready_agents[str(jid)] = True
                agent.logger.debug(f"AvailableCoordinatorState: {self.ready_agents}.")
                agent.logger.debug(f"Ready to subscribe message received from {jid}.")

        if self._are_all_agents_ready():
            for jid in self.ready_agents.keys():
                msg = Message(to=jid, sender=str(agent.jid.bare()))
                msg.body = "start to subscribe"
                msg.set_metadata("presence", "sync")
                await self.send(msg)

            agent.logger.debug(f"All start to subscribe messages sent.")
            self.set_next_state("subscription")

    def _are_all_agents_ready(self) -> bool:
        return all(self.ready_agents.values())


class SubscriptionCoordinatorState(State):

    def __init__(self, coordinated_agents: list[JID]):
        self.loop: int = 0
        self.ready_agents: dict[str, bool] = {
            str(jid.bare()): False for jid in coordinated_agents
        }
        super().__init__()

    async def on_start(self) -> None:
        if self.loop < 1:
            agent: CoordinatorAgent = self.agent
            agent.logger.debug(f"SubscriptionCoordinatorState: {self.ready_agents}.")
        self.loop += 1

    async def run(self) -> None:
        agent: CoordinatorAgent = self.agent
        if not self._are_all_agents_ready():
            msg = await self.receive(timeout=3)
            if msg and msg.body == "ready to start":
                jid = msg.sender.bare()
                self.ready_agents[str(jid)] = True
                agent.logger.debug(
                    f"SubscriptionCoordinatorState: {self.ready_agents}."
                )
                agent.logger.debug(f"Ready to start message received from {jid}.")

        if self._are_all_agents_ready():
            for jid in self.ready_agents.keys():
                msg = Message(to=jid, sender=str(agent.jid.bare()))
                msg.body = "start the algorithm"
                msg.set_metadata("presence", "sync")
                await self.send(msg)
                agent.logger.debug(f"All start the algorithm messages sent.")
                self.set_next_state("wait")
        else:
            self.set_next_state("subscription")

    def _are_all_agents_ready(self) -> bool:
        return all(self.ready_agents.values())


class WaitState(State):

    async def run(self) -> None:
        await asyncio.sleep(180)


class PresenceCoordinatorFSM(FSMBehaviour):

    def __init__(self, coordinated_agents: list[JID]):
        self.coordinated_agents = coordinated_agents
        super().__init__()

    async def on_start(self) -> None:
        agent: CoordinatorAgent = self.agent
        agent.logger.debug(f"PresenceCoordinatorFSM started.")

    def setup(self) -> None:
        self.add_state(
            name="available",
            state=AvailableCoordinatorState(self.coordinated_agents),
            initial=True,
        )
        self.add_state(
            name="subscription",
            state=SubscriptionCoordinatorState(self.coordinated_agents),
        )
        self.add_state(name="wait", state=WaitState())
        self.add_transition(source="available", dest="available")
        self.add_transition(source="available", dest="subscription")
        self.add_transition(source="subscription", dest="subscription")
        self.add_transition(source="subscription", dest="wait")

    async def on_end(self) -> Coroutine[Any, Any, None]:
        agent: CoordinatorAgent = self.agent
        agent.logger.debug(f"PresenceCoordinatorFSM behaviour finished.")
