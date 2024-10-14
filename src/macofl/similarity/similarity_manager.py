import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional

from aioxmpp import JID

from ..datatypes.models import ModelManager
from .similarity import SimilarityFunction
from .similarity_vector import SimilarityVector


class SimilarityManager:

    def __init__(
        self,
        model_manager: ModelManager,
        function: Optional[SimilarityFunction] = None,
    ) -> None:
        # NOTE Operations such as adding, removing, and reading a value on a dict are atomic.
        # Specifically:
        #     Adding a key and value mapping.
        #     Replacing a value for a key.
        #     Adding a dict to a dict via update().
        #     Getting a list of keys via keys().
        self.model_manager = model_manager
        self.function = function
        self.waiting_responses: list[JID] = []
        self.similarity_vectors: dict[JID, SimilarityVector | None] = (
            {}
        )  # the str is the UUID4 (or thread) and dict[JID, SimilarityVector | None] are the neighbours' vectors

    def get_own_similarity_vector(self) -> SimilarityVector:
        if self.function is None:
            raise ValueError(
                "The agent must have a function to compute the similarity vector."
            )
        layer2 = (
            self.model_manager.pretrain_state
            if self.model_manager.is_training()
            else self.model_manager.model.state_dict()
        )
        vector = self.function.get_similarity_vector(
            layers1=self.model_manager.initial_state,
            layers2=layer2,
        )
        vector.sent_time_z = datetime.now(tz=timezone.utc)
        return vector

    async def wait_similarity_vectors(self, timeout: Optional[float] = 10) -> None:
        start_time_z = datetime.now(tz=timezone.utc)
        stop_time_reached = False
        while self.waiting_responses and not stop_time_reached:
            await asyncio.sleep(delay=2)
            if timeout:
                stop_time_z = datetime.now(tz=timezone.utc) + timedelta(seconds=timeout)
                stop_time_reached = stop_time_z >= start_time_z

    def update_similarity_vector(
        self, neighbour: JID, vector: SimilarityVector
    ) -> None:
        if neighbour in self.waiting_responses:
            self.waiting_responses.remove(neighbour)
        self.similarity_vectors[neighbour] = vector

    def get_vector(self, neighbour: JID) -> SimilarityVector | None:
        if neighbour in self.similarity_vectors:
            return self.similarity_vectors[neighbour]
        return None
