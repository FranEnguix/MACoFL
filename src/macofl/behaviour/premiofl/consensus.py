import traceback
from typing import TYPE_CHECKING

from spade.behaviour import State

from ...datatypes.consensus_transmission import ConsensusTransmission
from ...message.message import RfMessage

if TYPE_CHECKING:
    from ...agent.premiofl.premiofl import PremioFlAgent


class ConsensusState(State):
    def __init__(self) -> None:
        self.agent: PremioFlAgent
        super().__init__()

    async def run(self) -> None:
        try:
            timeout = 10  # TODO: parametrizar timeout
            msg = await self.agent.receive(behaviour=self, timeout=timeout)

            if msg is None or RfMessage.is_completed(message=msg):
                if msg is None:
                    self.agent.logger.info(
                        f"[{self.agent.algorithm_iterations}] No message received in ConsensusState."
                    )
                elif RfMessage.is_completed(message=msg):
                    self.agent.message_logger.log(
                        iteration_id=self.agent.algorithm_iterations,
                        sender=msg.sender,
                        to=msg.to,
                        msg_type="RECV",
                        size=len(msg.body),
                    )
                    ct = ConsensusTransmission.from_message(message=msg)
                    self.agent.put_to_consensus_transmission_queue(ct)
                    self.agent.logger.info(
                        f"[{self.agent.algorithm_iterations}] Consensus message received in ConsensusState "
                        + f"from {ct.sender.localpart}."
                    )

                # Try to apply consensus
                self.agent.logger.debug(
                    f"[{self.agent.algorithm_iterations}] Starting consensus..."
                )
                cts = await self.agent.apply_all_consensus_transmission()
                if cts:
                    self.agent.logger.info(
                        f"[{self.agent.algorithm_iterations}] Consensus completed in ConsensusState with neighbours: "
                        + f"{[ct.sender.localpart for ct in cts]}."
                    )
                else:
                    self.agent.logger.debug(
                        f"[{self.agent.algorithm_iterations}] There are not consensus messages pending."
                    )
                self.set_next_state("train")

            elif RfMessage.is_multipart_and_not_yet_completed(message=msg):
                self.agent.logger.debug(
                    f"[{self.agent.algorithm_iterations}] Consensus received in ConsensusState from "
                    + f"{msg.sender.localpart} and waiting to rebuild multipart message to apply consensus."
                )
                self.set_next_state("consensus")

        except Exception as e:
            self.agent.logger.exception(e)
            traceback.print_exc()
