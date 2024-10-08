import traceback
from typing import TYPE_CHECKING

from spade.behaviour import State

from ...datatypes.consensus_transmission import ConsensusTransmission
from ...message.message import RfMessage

if TYPE_CHECKING:
    from ...agent.premiofl.acol import AcolAgent


class ConsensusReceiverState(State):
    def __init__(self) -> None:
        self.agent: AcolAgent
        super().__init__()

    async def run(self) -> None:
        try:
            timeout = 10  # TODO: parametrizar timeout
            msg = await self.agent.receive(behaviour=self, timeout=timeout)

            if msg is None:
                self.agent.logger.info(
                    f"[{self.agent.algorithm_iterations}] No message in ReceiverState... going to Train."
                )
                self.set_next_state("train")
            elif RfMessage.is_completed(message=msg):
                ct = ConsensusTransmission.from_message(message=msg)
                cts = await self.agent.apply_all_consensus_transmission(
                    consensus_transmission=ct, send_model_during_consensus=False
                )
                self.agent.logger.info(
                    f"[{self.agent.algorithm_iterations}] Consensus received in ReceiverState from {ct.sender.localpart} and consensus applied with neighbours: {[ct.sender.localpart for ct in cts]}."
                )
                self.agent.message_logger.log(
                    iteration_id=self.agent.algorithm_iterations,
                    sender=msg.sender,
                    to=msg.to,
                    msg_type="RECV",
                    size=len(msg.body),
                )
                self.set_next_state("train")
            elif RfMessage.is_multipart_and_not_yet_completed(message=msg):
                self.agent.logger.debug(
                    f"[{self.agent.algorithm_iterations}] Consensus received in ReceiverState from {msg.sender.localpart} and waiting to rebuild message to apply consensus."
                )
                self.set_next_state("receive")
        except Exception as e:
            self.agent.logger.exception(e)
            traceback.print_exc()
