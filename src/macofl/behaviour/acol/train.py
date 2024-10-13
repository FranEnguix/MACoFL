import traceback
from typing import TYPE_CHECKING

from spade.behaviour import State

from ...datatypes.metrics import ModelMetrics

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
                metrics_train = self.agent.model_manager.train(
                    train_logger=self.agent.nn_train_logger.log_train_epoch,
                    agent_jid=self.agent.jid,
                    algorithm_iteration=self.agent.algorithm_iterations,
                )

                metrics_validation = self.agent.model_manager.inference()
                metrics_test = self.agent.model_manager.test_inference()
                self.log_inference_results(
                    trains=metrics_train,
                    validation=metrics_validation,
                    test=metrics_test,
                )

                # Apply consensus
                consensus_transmissions_applied = (
                    await self.agent.apply_all_consensus_transmission(
                        send_model_during_consensus=False
                    )
                )
                self.agent.logger.info(
                    f"[{self.agent.algorithm_iterations}] Consensus completed with neighbours: "
                    + f"{[ct.sender.localpart for ct in consensus_transmissions_applied]}."
                )
                self.set_next_state("send")

        except Exception as e:
            self.agent.logger.exception(e)
            traceback.print_exc()

    def log_inference_results(
        self, trains: list[ModelMetrics], validation: ModelMetrics, test: ModelMetrics
    ) -> None:
        if trains:
            start_t = trains[0].start_time_z
            end_t = trains[-1].end_time_z
            if start_t is not None and end_t is not None:
                train_time = end_t - start_t
                mean_accuracy = sum(m.accuracy for m in trains) / len(trains)
                mean_loss = sum(m.accuracy for m in trains) / len(trains)
                self.agent.logger.info(
                    f"[{self.agent.algorithm_iterations}] Train ({len(trains)} epochs) completed in "
                    + f"{train_time.total_seconds():.2f} seconds with mean accuracy {mean_accuracy} and mean"
                    + f" loss {mean_loss}."
                )
                self.agent.nn_inference_logger.log(
                    iteration_id=self.agent.algorithm_iterations,
                    agent=self.agent.jid,
                    seconds=train_time.total_seconds(),
                    epochs=len(trains),
                    mean_training_accuracy=mean_accuracy,
                    mean_training_loss=mean_loss,
                    validation_accuracy=validation.accuracy,
                    validation_loss=validation.loss,
                    test_accuracy=test.accuracy,
                    test_loss=test.loss,
                    timestamp=start_t,
                )
