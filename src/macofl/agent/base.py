import logging
import traceback
from typing import Optional

from aioxmpp import JID, Presence, PresenceType
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message
from spade.template import Template

from ..behaviour.coordination import PresenceNodeFSM
from ..message.message import RfMessage
from ..message.multipart import MultipartHandler


class AgentBase(Agent):
    def __init__(
        self,
        jid: str,
        password: str,
        max_message_size: int,
        web_address: str = "0.0.0.0",
        web_port: int = 10000,
        verify_security: bool = False,
    ):
        name = JID.fromstr(jid).localpart
        self.logger = logging.getLogger(f"rf.log.agent.{name}")
        self.message_logger = logging.getLogger(f"rf.message.agent.{name}")
        self.accuracy_logger = logging.getLogger(f"rf.accuracy.agent.{name}")
        self.max_message_size = max_message_size
        self.web_address = web_address
        self.web_port = web_port
        self._multipart_handler = MultipartHandler()
        super().__init__(jid=jid, password=password, verify_security=verify_security)

    async def setup(self) -> None:
        self.setup_presence_handlers()

    async def send(
        self, message: Message, behaviour: Optional[CyclicBehaviour] = None
    ) -> None:
        messages = self._multipart_handler.generate_multipart_messages(
            content=message.body,
            max_size=self.max_message_size,
            message_base=message,
        )
        if messages is None:
            messages = [message]
        for msg in messages:
            if behaviour is not None:
                await behaviour.send(msg=msg)
            else:
                futures = self.dispatch(msg=msg)
                for f in futures:
                    f.result()
            self.logger.debug(
                f"Message ({msg.sender.bare()}) -> ({msg.to.bare()}): {msg.body}"
            )

    async def receive(
        self, behaviour: CyclicBehaviour, timeout: Optional[float] = 0
    ) -> RfMessage | None:
        """
        Put a behaviour start listening to messages using the MultipartHandler class.
        If a message arrives, this function returns the message, otherwise returns None.
        If the message is a multipart message and it is completed, returns the completed
        message with the flags `is_multipart` and `is_multipart_completed` set to True.

        Args:
            behaviour (CyclicBehaviour): The receiver behaviour.
            timeout (Optional[float], optional): Timeout in seconds. Defaults to 0.

        Returns:
            RfMessage | None: The message received -and rebuilded if necessary- or None
            if any message arrived.
        """
        msg: Message | None = await behaviour.receive(timeout=timeout)
        if msg is not None:
            is_multipart = self._multipart_handler.is_multipart(msg)
            if is_multipart:
                header = self._multipart_handler.get_header(msg.body)
                self.logger.debug(
                    f"Multipart message arrived from {msg.sender}: {header} with length {len(msg.body)}"
                )
                multipart_msg = self._multipart_handler.rebuild_multipart(message=msg)
                is_multipart_completed = multipart_msg is not None
                if is_multipart_completed:
                    return RfMessage.from_message(
                        message=multipart_msg,
                        is_multipart=is_multipart,
                        is_multipart_completed=is_multipart_completed,
                    )
                return RfMessage.from_message(
                    message=msg,
                    is_multipart=is_multipart,
                    is_multipart_completed=is_multipart_completed,
                )
            self.logger.debug(
                f"Message arrived from {msg.sender}: with length {len(msg.body)}"
            )
            return RfMessage.from_message(
                message=msg, is_multipart=False, is_multipart_completed=False
            )
        return None

    def any_multipart_waiting(self) -> bool:
        return self._multipart_handler.any_multipart_waiting()

    def on_available(self, jid: str, stanza) -> None:
        self.logger.debug(f"{jid} is available with stanza {stanza}.")

    def on_subscribed(self, jid) -> None:
        self.logger.debug(f"{jid} has accepted my subscription request.")
        self.logger.debug(f"My contact list is {self.presence.get_contacts()}.")

    def on_subscribe(self, jid) -> None:
        self.presence.approve(jid)
        self.logger.debug(f"{jid} approved.")

    def setup_presence_handlers(self) -> None:
        self.presence.on_subscribe = self.on_subscribe
        self.presence.on_subscribed = self.on_subscribed
        self.presence.on_available = self.on_available


class AgentNodeBase(AgentBase):
    def __init__(
        self,
        jid: str,
        password: str,
        max_message_size: int,
        observers: Optional[list[JID]] = None,
        neighbours: Optional[list[JID]] = None,
        coordinator: Optional[JID] = None,
        post_coordination_behaviours: Optional[
            list[tuple[CyclicBehaviour, Template]]
        ] = None,
        web_address: str = "0.0.0.0",
        web_port: int = 10000,
        verify_security: bool = False,
    ):
        self.observers = [] if observers is None else observers
        self.neighbours = [] if neighbours is None else neighbours
        self.coordinator = coordinator
        self.post_coordination_behaviours = (
            [] if post_coordination_behaviours is None else post_coordination_behaviours
        )
        self.coordination_fsm: PresenceNodeFSM | None = None
        super().__init__(
            jid=jid,
            password=password,
            max_message_size=max_message_size,
            web_address=web_address,
            web_port=web_port,
            verify_security=verify_security,
        )

    async def setup(self) -> None:
        await super().setup()
        if self.coordinator is not None:
            self.coordination_fsm = PresenceNodeFSM(self.coordinator)
            template = Template()
            template.set_metadata("rf.presence", "sync")
            self.add_behaviour(self.coordination_fsm, template)
            self.logger.info("PresenceNodeFSM attached.")
        else:
            self.logger.info("Starting without PresenceNodeFSM.")
            for behaviour, template in self.post_coordination_behaviours:
                self.add_behaviour(behaviour, template)

    def subscribe_to_neighbours(self) -> None:
        try:
            for jid in self.neighbours:
                self.presence.subscribe(str(jid.bare()))
                self.logger.debug(f"Subscription request sent to {jid}")
        except Exception:
            traceback.print_exc()

    def get_non_subscribe_both_neighbours(self) -> dict[JID, str]:
        contacts: dict[JID, dict] = self.presence.get_contacts()
        result = {
            j.bare(): data["subscription"]
            for j, data in contacts.items()
            if j in self.neighbours and data["subscription"] != "both"
        }
        for jid in self.neighbours:
            if jid.bare() not in result.keys():
                result[jid.bare()] = "null"
        return result

    def is_presence_completed(self) -> bool:
        contacts: dict[JID, dict] = self.presence.get_contacts()
        if not all(ag.bare() in contacts for ag in self.neighbours):
            return False
        return all(data["subscription"] == "both" for data in contacts.values())

    def get_available_neighbours(self) -> list[JID]:
        # TODO: check if neighbour is available with self.presence.get_contacts()
        return self.neighbours
