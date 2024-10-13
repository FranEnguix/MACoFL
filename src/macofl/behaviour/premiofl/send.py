import traceback
import uuid
from typing import TYPE_CHECKING, Optional, OrderedDict

from aioxmpp import JID
from spade.behaviour import State
from torch import Tensor

from ...similarity.similarity_vector import SimilarityVector

if TYPE_CHECKING:
    from ...agent.premiofl.premiofl import PremioFlAgent


class SendState(State):
    def __init__(self) -> None:
        self.agent: PremioFlAgent
        super().__init__()

    async def on_start(self) -> None:
        self.agent.logger.debug(
            f"[{self.agent.algorithm_iterations}] Starting SendState..."
        )

    async def run(self) -> None:
        try:
            selected_neighbours = self.agent.select_neighbours()
            if selected_neighbours:
                self.agent.logger.info(
                    f"[{self.agent.algorithm_iterations}] Selected neighbours of SendState: "
                    + f"{[jid.localpart for jid in selected_neighbours]}"
                )

                thread = str(uuid.uuid4())
                if self.agent.similarity_manager.function is not None:
                    await self.similarity_vector_exchange(
                        selected_neighbours, thread=thread
                    )
                for n, ls in self.agent.assign_layers(selected_neighbours):
                    await self.send_layers(neighbour=n, layers=ls, thread=thread)

                self.agent.logger.info(
                    f"[{self.agent.algorithm_iterations}] Selected layers sent to selected neighbours."
                )
                self.set_next_state("consensus")

            else:
                self.agent.logger.info(
                    f"[{self.agent.algorithm_iterations}] No neighbour selected in SendState. Going to train again..."
                )
                self.set_next_state("train")

        except Exception as e:
            self.agent.logger.exception(e)
            traceback.print_exc()

    async def send_layers(
        self,
        neighbour: JID,
        layers: OrderedDict[str, Tensor],
        thread: Optional[str] = None,
    ) -> None:
        metadata = {"rf.conversation": "layers"}
        await self.agent.send_local_layers(
            neighbour=neighbour,
            layers=layers,
            thread=thread,
            metadata=metadata,
            behaviour=self,
        )
        self.agent.logger.debug(
            f"[{self.agent.algorithm_iterations}] Sent to {neighbour.localpart} the layers: {list(layers.keys())}."
        )

    async def similarity_vector_exchange(
        self, neighbours: list[JID], thread: str
    ) -> None:
        vector = self.agent.similarity_manager.get_own_similarity_vector()
        vector.owner = self.agent.jid
        for neighbour in neighbours:
            await self.send_similarity_vector(
                uuid4=thread, vector=vector, neighbour=neighbour
            )
        pending_agents = await self.agent.similarity_manager.wait_similarity_vectors(
            uuid4=thread, timeout=600
        )

    async def send_similarity_vector(
        self,
        uuid4: str,
        vector: SimilarityVector,
        neighbour: JID,
    ) -> None:
        metadata = {"rf.conversation": "similarity"}
        await self.agent.send_similarity_vector(
            neighbour=neighbour,
            vector=vector,
            thread=uuid4,
            metadata=metadata,
            behaviour=self,
        )
        self.agent.logger.debug(
            f"[{self.agent.algorithm_iterations}] Sent to {neighbour.localpart} the vector: {vector.to_message().body} "
            + f"with thread {uuid4}."
        )
