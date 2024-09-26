import asyncio
import logging
import traceback
from typing import Coroutine, Any, TYPE_CHECKING

from aioxmpp import JID
from spade.behaviour import FSMBehaviour, State, CyclicBehaviour
from spade.message import Message

if TYPE_CHECKING:
    from macofl.agent.coordinator import CoordinatorAgent
    from macofl.agent import AgentNodeBase


class ObserverBehaviour(CyclicBehaviour):

    def __init__(self, logger_name: str):
        self.logger = logging.getLogger(logger_name)
        super().__init__()

    async def run(self) -> None:
        await asyncio.sleep(0.1)
        msg = await self.receive(1)
        if msg:
            self.logger.info(msg.body)
