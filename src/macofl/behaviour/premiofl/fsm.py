from typing import TYPE_CHECKING

from spade.behaviour import FSMBehaviour

from .consensus import ConsensusState
from .send import SendState
from .train import TrainAndApplyConsensusState

if TYPE_CHECKING:
    from ...agent.premiofl.premiofl import PremioFlAgent


class AcolFsmBehaviour(FSMBehaviour):

    def __init__(self) -> None:
        self.agent: PremioFlAgent
        self.train_state = TrainAndApplyConsensusState()
        self.send_state = SendState()
        self.consensus_state = ConsensusState()
        super().__init__()

    def setup(self) -> None:
        self.add_state(name="train", state=self.train_state, initial=True)
        self.add_state(name="send", state=self.send_state)
        self.add_state(name="consensus", state=self.consensus_state)
        self.add_transition(source="train", dest="send")
        self.add_transition(source="send", dest="train")
        self.add_transition(source="send", dest="consensus")
        self.add_transition(source="consensus", dest="consensus")
        self.add_transition(source="consensus", dest="train")

    async def on_start(self) -> None:
        self.agent.logger.debug("FSM of ACoL algorithm started.")

    async def on_end(self) -> None:
        self.agent.logger.debug("FSM of ACoL algorithm finished.")
