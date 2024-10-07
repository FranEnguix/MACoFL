from typing import TYPE_CHECKING

from spade.behaviour import FSMBehaviour

from .receive import ConsensusReceiverState
from .send import SendState
from .train import TrainAndApplyConsensusState

if TYPE_CHECKING:
    from ...agent.premiofl.premiofl import PremioFlAgent


class AcolFsmBehaviour(FSMBehaviour):

    def __init__(self) -> None:
        self.agent: PremioFlAgent
        self.train_state = TrainAndApplyConsensusState()
        self.send_state = SendState()
        self.receive_state = ConsensusReceiverState()
        super().__init__()

    def setup(self) -> None:
        self.add_state(name="train", state=self.train_state, initial=True)
        self.add_state(name="send", state=self.send_state)
        self.add_state(name="receive", state=self.receive_state)
        self.add_transition(source="train", dest="send")
        self.add_transition(source="send", dest="train")
        self.add_transition(source="send", dest="receive")
        self.add_transition(source="receive", dest="receive")
        self.add_transition(source="receive", dest="train")

    async def on_start(self) -> None:
        self.agent.logger.debug("FSM of ACoL algorithm started.")

    async def on_end(self) -> None:
        self.agent.logger.debug("FSM of ACoL algorithm finished.")
