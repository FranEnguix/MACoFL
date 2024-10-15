from typing import TYPE_CHECKING

from spade.behaviour import State

# from ...message.message import RfMessage

if TYPE_CHECKING:
    from ...agent.premiofl.premiofl import PremioFlAgent


class ConsensusState(State):
    def __init__(self) -> None:
        self.agent: PremioFlAgent
        super().__init__()

    async def run(self) -> None:
        consensus_it_id = (
            self.agent.consensus_manager.get_completed_iterations(
                self.agent.current_round
            )
            + 1
        )
        self.agent.logger.debug(
            f"[{self.agent.current_round}] Waiting for layers to apply consensus..."
        )
        if await self.agent.consensus_manager.wait_receive_consensus():
            self.agent.logger.info(
                f"[{self.agent.current_round}] ({consensus_it_id}) All layers received."
            )
        else:
            self.agent.logger.debug(
                f"[{self.agent.current_round}] Receive consensus finished by timeout."
            )
        # Try to apply consensus
        self.agent.logger.debug(f"[{self.agent.current_round}] Starting consensus...")
        consensuateds = self.agent.consensus_manager.apply_all_consensus()
        if consensuateds:
            self.agent.logger.info(
                f"[{self.agent.current_round}] ({consensus_it_id}) Consensus completed in ConsensusState "
                + f"with neighbours: {[ct.sender.localpart for ct in consensuateds if ct.sender]}."
            )
        else:
            self.agent.logger.debug(
                f"[{self.agent.current_round}] There are not consensus messages pending to consensuate."
            )

    async def on_end(self):
        it = self.agent.consensus_manager.add_one_completed_iteration(
            algorithm_rounds=self.agent.current_round
        )
        if self.agent.consensus_manager.are_max_iterations_reached():
            self.agent.logger.info(
                f"[{self.agent.current_round}] Going to TrainState because max consensus iterations "
                + f"reached: {it}/{self.agent.consensus_manager.max_iterations}."
            )
            self.set_next_state("train")
        else:
            self.agent.logger.info(
                f"[{self.agent.current_round}] Going to CommunicationState... iterations: "
                + f"{it}/{self.agent.consensus_manager.max_iterations}."
            )
            self.set_next_state("communication")
