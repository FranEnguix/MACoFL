from typing import TYPE_CHECKING

from spade.behaviour import State

if TYPE_CHECKING:
    from ...agent.premiofl.acol import AcolAgent

import traceback


class SendState(State):
    def __init__(self) -> None:
        self.agent: AcolAgent
        super().__init__()

    async def on_start(self) -> None:
        self.agent.logger.debug(
            f"[{self.agent.algorithm_iterations}] Starting SendState..."
        )

    async def run(self) -> None:
        try:
            selected_neighbours = self.agent.select_neighbours()
            self.agent.logger.info(
                f"[{self.agent.algorithm_iterations}] Selected neighbours of SendState: {[jid.localpart for jid in selected_neighbours]}"
            )
            if selected_neighbours:
                metadata = {"rf.conversation": "consensus_send_to_cyclic_receive"}
                for neighbour in selected_neighbours:
                    await self.agent.send_local_weights(
                        neighbour=neighbour, metadata=metadata, behaviour=self
                    )
                    self.agent.logger.info(
                        f"[{self.agent.algorithm_iterations}] Local weights sent to: {neighbour.localpart}"
                    )
                self.set_next_state("receive")
            else:
                self.set_next_state("train")
        except Exception as e:
            self.agent.logger.exception(e)
            traceback.print_exc()
