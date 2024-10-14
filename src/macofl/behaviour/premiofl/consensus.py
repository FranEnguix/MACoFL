import traceback
from typing import TYPE_CHECKING, Optional, OrderedDict

from aioxmpp import JID
from spade.behaviour import State
from torch import Tensor

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
            # Try to apply consensus
            self.agent.logger.debug(
                f"[{self.agent.algorithm_iterations}] Starting consensus..."
            )
            cts = await self.agent.apply_all_consensus_transmission()
            if cts:
                for ct in cts:
                    if ct.request_reply:
                        model = self.agent.model_manager.get_layers(
                            layers=list(ct.model.keys())
                        )
                        await self.send_layers(neighbour=ct.sender, layers=model)

                self.agent.logger.info(
                    f"[{self.agent.algorithm_iterations}] Consensus completed in ConsensusState with neighbours: "
                    + f"{[ct.sender.localpart for ct in cts]}."
                )
            else:
                self.agent.logger.debug(
                    f"[{self.agent.algorithm_iterations}] There are not consensus messages pending."
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
            request_reply=True,
            layers=layers,
            thread=thread,
            metadata=metadata,
            behaviour=self,
        )
        self.agent.logger.debug(
            f"[{self.agent.algorithm_iterations}] Sent to {neighbour.localpart} the layers: {list(layers.keys())}."
        )
