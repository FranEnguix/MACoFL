import copy
from queue import Queue
from typing import OrderedDict

from torch import Tensor


class Consensus:

    def __init__(
        self,
        max_order: int,
        max_seconds_to_accept_pre_consensus: float,
        epsilon_margin: float = 0.05,
    ) -> None:
        self.models_to_consensuate: Queue[OrderedDict[str, Tensor]] = Queue()
        self.max_order = max_order
        self.max_seconds_to_accept_pre_consensus = max_seconds_to_accept_pre_consensus
        self.epsilon_margin = epsilon_margin

    def apply_all_consensus(
        self, model: OrderedDict[str, Tensor]
    ) -> OrderedDict[str, Tensor]:
        consensuated_model = copy.deepcopy(model)
        while self.models_to_consensuate.qsize() > 0:
            weights_and_biases = self.models_to_consensuate.get()
            consensuated_model = Consensus.apply_consensus(
                consensuated_model,
                weights_and_biases,
                max_order=self.max_order,
                epsilon_margin=self.epsilon_margin,
            )
            self.models_to_consensuate.task_done()
        return consensuated_model

    @staticmethod
    def apply_consensus(
        weights_and_biases_a: OrderedDict[str, Tensor],
        weights_and_biases_b: OrderedDict[str, Tensor],
        max_order: int = 2,
        epsilon_margin: float = 0.05,
    ) -> OrderedDict[str, Tensor]:
        consensuated_result: OrderedDict[str, Tensor] = OrderedDict()
        for key in weights_and_biases_a.keys():
            if key in weights_and_biases_b:
                consensuated_result[key] = Consensus.consensus_update_to_tensors(
                    tensor_a=weights_and_biases_a[key],
                    tensor_b=weights_and_biases_b[key],
                    max_order=max_order,
                    epsilon_margin=epsilon_margin,
                )
            else:
                raise ValueError(
                    f"Consensus error. The key '{key}' is not present in both models."
                )
        return consensuated_result

    @staticmethod
    def consensus_update_to_tensors(
        tensor_a: Tensor, tensor_b: Tensor, max_order: int, epsilon_margin: float = 0.05
    ) -> Tensor:
        if max_order <= 1:
            raise ValueError(
                f"Max order of consensus must be greater than 1 and get {max_order}."
            )
        # epsilon_margin because must be LESS than 1 / max_order
        epsilon = 1 / max_order - epsilon_margin
        return epsilon * tensor_a + (1 - epsilon) * tensor_b
