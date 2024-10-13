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
        self.similarity_vectors: dict[str, dict[JID, SimilarityVector | None]] = (
            {}
        )  # the str is the UUID4 (or thread) and dict[JID, SimilarityVector | None] are the neighbours' vectors

    def add_waiting_response_flag(self, uuid4: str, neighbour: JID) -> None:
        if uuid4 not in self.similarity_vectors:
            self.similarity_vectors[uuid4] = {}
        self.similarity_vectors[uuid4][neighbour] = None

    def get_own_similarity_vector(self) -> SimilarityVector:
        if self.function is None:
            raise ValueError(
                "The agent must have a function to compute the similarity vector."
            )
        vector = self.function.get_similarity_vector(
            layers1=self.model_manager.initial_state,
            layers2=self.model_manager.model.state_dict(),
        )
        vector.sent_time_z = datetime.now(tz=timezone.utc)
        return vector

    async def wait_similarity_vectors(
        self, uuid4: str, timeout: Optional[float] = 10
    ) -> list[JID]:
        thread = self.similarity_vectors[uuid4]
        waiting = [neighbour for neighbour, vector in thread.items() if not vector]
        start_time_z = datetime.now(tz=timezone.utc)
        stop_time_reached = False
        while waiting and not stop_time_reached:
            await asyncio.sleep(delay=2)
            waiting = [neighbour for neighbour, vector in thread.items() if not vector]
            if timeout:
                stop_time_z = datetime.now(tz=timezone.utc) + timedelta(seconds=timeout)
                stop_time_reached = stop_time_z >= start_time_z
        return waiting

    def i_have_to_answer_thread(self, uuid: str) -> bool:
        return not uuid in self.similarity_vectors

    def update_similarity_vector(
        self, uuid4: str, neighbour: JID, vector: SimilarityVector
    ) -> None:
        if uuid4 not in self.similarity_vectors:
            self.similarity_vectors[uuid4] = {}
        self.similarity_vectors[uuid4][neighbour] = vector

    def get_vector(self, uuid4: str, neighbour: JID) -> SimilarityVector | None:
        if (
            uuid4 in self.similarity_vectors
            and neighbour in self.similarity_vectors[uuid4]
        ):
            return self.similarity_vectors[uuid4][neighbour]
        return None
