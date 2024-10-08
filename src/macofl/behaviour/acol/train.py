import traceback
from typing import TYPE_CHECKING

from spade.behaviour import State

if TYPE_CHECKING:
    from ...agent.premiofl.premiofl import PremioFlAgent


class TrainAndApplyConsensusState(State):
    def __init__(self) -> None:
        # self.iterations: int = 0
        self.agent: PremioFlAgent
        super().__init__()

    async def on_start(self) -> None:
        self.agent.algorithm_iterations += 1
        if self.agent.are_max_iterations_reached():
            self.agent.logger.info(
                f"[{self.agent.algorithm_iterations - 1}] Stopping agent because max_algorithm_iterations reached: {self.agent.algorithm_iterations - 1}/{self.agent.max_algorithm_iterations}"
            )
            await self.agent.stop()
        else:
            self.agent.logger.info(
                f"[{self.agent.algorithm_iterations}] Starting ACoL algorithm iteration id: {self.agent.algorithm_iterations}"
            )

    async def run(self) -> None:
        try:
            if not self.agent.are_max_iterations_reached():
                # Train the model
                metrics_train = self.agent.model_manager.train()
                metrics_test = self.agent.model_manager.test_inference()
                self.agent.logger.info(
                    f"[{self.agent.algorithm_iterations}] Train completed in {metrics_train.time_elapsed().total_seconds():.2f} seconds with accuracy {metrics_train.accuracy} and loss {metrics_train.loss}."
                )
                self.agent.nn_logger.log(
                    iteration_id=self.agent.algorithm_iterations,
                    agent=self.agent.jid,
                    seconds=metrics_train.time_elapsed().seconds,
                    training_accuracy=metrics_train.accuracy,
                    training_loss=metrics_train.loss,
                    test_accuracy=metrics_test.accuracy,
                    test_loss=metrics_test.loss,
                )

                # Apply consensus
                cts = await self.agent.apply_all_consensus_transmission(
                    send_model_during_consensus=False
                )
                self.agent.logger.info(
                    f"[{self.agent.algorithm_iterations}] Consensus completed with neighbours: {[ct.sender.localpart for ct in cts]}."
                )
                self.set_next_state("send")
        except Exception as e:
            self.agent.logger.exception(e)
            traceback.print_exc()
