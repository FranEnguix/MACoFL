from datetime import datetime, timezone
from typing import TYPE_CHECKING

from spade.behaviour import CyclicBehaviour

from ...datatypes.consensus_transmission import ConsensusTransmission

if TYPE_CHECKING:
    from ...agent.premiofl.premiofl import PremioFlAgent


class CyclicConsensusReceiverBehaviour(CyclicBehaviour):

    def __init__(self) -> None:
        super().__init__()
        self.agent: PremioFlAgent

    async def run(self) -> None:
        msg = await self.agent.receive(self, timeout=4)
        if msg:
            # TODO: log message received
            consensus_t = ConsensusTransmission.from_message(message=msg)
            consensus_t.received_time_z = datetime.now(tz=timezone.utc)  # zulu = utc+0

            if not consensus_t.sent_time_z:
                raise ValueError(
                    f"Consensus message from {msg.sender.bare()} without timestamp."
                )

            seconds_since_message_sent = (
                consensus_t.received_time_z - consensus_t.sent_time_z
            )
            max_seconds_pre_consensus = (
                self.agent.consensus.max_seconds_to_accept_pre_consensus
            )

            if seconds_since_message_sent.total_seconds() <= max_seconds_pre_consensus:
                # TODO: log pre consensus accepted

                self.agent.put_to_consensus_transmission_queue(
                    consensus_transmission=consensus_t
                )
            else:
                pass  # TODO: log mensaje descartado por exceso de tiempo
