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
            consensus_tr = ConsensusTransmission.from_message(message=msg)
            consensus_tr.received_time_z = datetime.now(tz=timezone.utc)  # zulu = utc+0

            if not consensus_tr.sent_time_z:
                error_msg = (
                    f"[{self.agent.algorithm_iterations}] Consensus message from {msg.sender.bare()} without "
                    + "timestamp."
                )
                self.agent.logger.exception(error_msg)
                raise ValueError(error_msg)

            seconds_since_message_sent = (
                consensus_tr.received_time_z - consensus_tr.sent_time_z
            )
            max_seconds_pre_consensus = (
                self.agent.consensus.max_seconds_to_accept_pre_consensus
            )

            if seconds_since_message_sent.total_seconds() <= max_seconds_pre_consensus:
                self.agent.logger.info(
                    f"[{self.agent.algorithm_iterations}] Consensus message accepted in ReceiverBehaviour with time "
                    + f"elapsed {seconds_since_message_sent.total_seconds():.2f}"
                )
                self.agent.put_to_consensus_transmission_queue(
                    consensus_transmission=consensus_tr
                )
            else:
                self.agent.logger.info(
                    f"[{self.agent.algorithm_iterations}] Consensus message discarted in ReceiverBehaviour because time"
                    + f" elapsed is {seconds_since_message_sent.total_seconds():.2f} and maximum is "
                    + f"{max_seconds_pre_consensus:.2f}"
                )
