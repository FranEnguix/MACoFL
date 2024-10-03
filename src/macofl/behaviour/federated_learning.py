from datetime import datetime, timezone
from typing import TYPE_CHECKING

from spade.behaviour import CyclicBehaviour

from ..message.message import MessageFl

if TYPE_CHECKING:
    from ..agent.federated_learning.premiofl import PremioFlAgent


class ReceiveWeights(CyclicBehaviour):

    def __init__(self) -> None:
        super().__init__()
        self.agent: PremioFlAgent

    async def on_end(self) -> None:
        return await super().on_end()

    async def run(self) -> None:
        super().run()
        msg = await self.agent.receive(self, timeout=4)
        if msg:
            # TODO: log message received
            message = MessageFl(msg)
            now_z = datetime.now(tz=timezone.utc)  # zulu = utc+0
            msg_timestamp_z = message.timestamp_z

            seconds_since_message_sent = now_z - msg_timestamp_z
            max_seconds_pre_consensus = (
                self.agent.consensus.max_seconds_to_accept_pre_consensus
            )

            if seconds_since_message_sent.total_seconds() <= max_seconds_pre_consensus:
                # TODO: log pre consensus accepted

                self.agent.store_consensus_weights(message.weights)
                if not self.agent.is_training():
                    self.agent.apply_consensus()

                # TODO: enviar la matriz consensuada (o no) a un agente aleatorio
                await self.agent.send_local_weights()
            else:
                pass  # TODO: log mensaje descartado por exceso de tiempo
