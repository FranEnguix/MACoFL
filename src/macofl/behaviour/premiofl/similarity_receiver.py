from datetime import datetime, timezone
from typing import TYPE_CHECKING

from spade.behaviour import CyclicBehaviour

from ...message.message import RfMessage
from ...similarity.similarity_vector import SimilarityVector

if TYPE_CHECKING:
    from ...agent.premiofl.premiofl import PremioFlAgent


class SimilarityReceiverBehaviour(CyclicBehaviour):

    def __init__(self) -> None:
        self.agent: PremioFlAgent
        super().__init__()

    async def run(self) -> None:
        timeout = 2  # TODO parametrizar
        msg = await self.agent.receive(self, timeout=timeout)
        if (
            msg
            and RfMessage.is_completed(message=msg)
            and not self.agent.are_max_iterations_reached()
        ):
            vector = SimilarityVector.from_message(message=msg)
            vector.received_time_z = datetime.now(tz=timezone.utc)  # zulu = utc+0

            if not vector.sent_time_z:
                error_msg = (
                    f"[{self.agent.algorithm_iterations}] Similarity vector from {msg.sender.bare()} without "
                    + "timestamp."
                )
                self.agent.logger.exception(error_msg)
                raise ValueError(error_msg)

            self.agent.similarity_manager.update_similarity_vector(
                uuid4=msg.thread, neighbour=msg.sender, vector=vector
            )

            seconds_since_message_sent = vector.received_time_z - vector.sent_time_z
            self.agent.logger.info(
                f"[{self.agent.algorithm_iterations}] Similarity vector ({msg.thread}) received from "
                + f"{msg.sender.bare()} in SimilarityReceiverBehaviour with time elapsed "
                + f"{seconds_since_message_sent.total_seconds():.2f}"
            )
