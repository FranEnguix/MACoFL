from datetime import datetime, timezone
from typing import TYPE_CHECKING

from spade.behaviour import CyclicBehaviour

from ...datatypes.consensus_transmission import ConsensusTransmission
from ...message.message import RfMessage

if TYPE_CHECKING:
    from ...agent.premiofl.premiofl import PremioFlAgent


class LayerReceiverBehaviour(CyclicBehaviour):

    def __init__(self) -> None:
        self.agent: PremioFlAgent
        super().__init__()

    async def run(self) -> None:
        timeout = 5  # TODO parametrizar
        msg = await self.agent.receive(self, timeout=timeout)
        if (
            msg
            and RfMessage.is_completed(message=msg)
            and not self.agent.are_max_iterations_reached()
        ):
            self.agent.message_logger.log(
                iteration_id=self.agent.algorithm_iterations,
                sender=msg.sender,
                to=msg.to,
                msg_type="RECV",
                size=len(msg.body),
            )
            consensus_tr = ConsensusTransmission.from_message(message=msg)
            consensus_tr.received_time_z = datetime.now(tz=timezone.utc)  # zulu = utc+0

            if not consensus_tr.sent_time_z:
                error_msg = (
                    f"[{self.agent.algorithm_iterations}] Consensus message from {msg.sender.bare()} without "
                    + "timestamp."
                )
                self.agent.logger.exception(error_msg)
                raise ValueError(error_msg)

            time_elapsed = consensus_tr.received_time_z - consensus_tr.sent_time_z
            max_seconds_consensus = self.agent.consensus.max_seconds_to_accept_consensus

            if time_elapsed.total_seconds() <= max_seconds_consensus:
                self.agent.logger.info(
                    f"[{self.agent.algorithm_iterations}] Consensus message accepted in LayerReceiverBehaviour with "
                    + f"time elapsed {time_elapsed.total_seconds():.2f}"
                )
                self.agent.put_to_consensus_transmission_queue(
                    consensus_transmission=consensus_tr
                )
            else:
                self.agent.logger.info(
                    f"[{self.agent.algorithm_iterations}] Consensus message discarted in LayerReceiverBehaviour because"
                    + f" time elapsed is {time_elapsed.total_seconds():.2f} and maximum is {max_seconds_consensus:.2f}"
                )
