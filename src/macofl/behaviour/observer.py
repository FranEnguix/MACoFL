import asyncio
import traceback
from typing import Coroutine, Any, TYPE_CHECKING

from aioxmpp import JID
from spade.behaviour import FSMBehaviour, State, CyclicBehaviour
from spade.message import Message

if TYPE_CHECKING:
    from macofl.agent.coordinator import CoordinatorAgent
    from macofl.agent import AgentNodeBase


class ObserveBehaviour(CyclicBehaviour):

    def __init__(
        self,
    ):
        super().__init__()
