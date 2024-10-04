import copy
from queue import Queue
from typing import OrderedDict

from torch import Tensor


class Consensus:

    def __init__(
        self,
        epsilon: int,
        max_order: int,
        max_seconds_to_accept_pre_consensus: float,
    ) -> None:
        self.models_to_consensuate: Queue[OrderedDict[str, Tensor]] = Queue()
        self.epsilon = epsilon
        self.max_order = max_order
        if epsilon <= 0 or epsilon >= 1:
            raise RuntimeError("Consensus epsilon must be in (0, 1) range.")
        self.max_seconds_to_accept_pre_consensus = max_seconds_to_accept_pre_consensus

    def apply_all_consensus(
        self, model: OrderedDict[str, Tensor]
    ) -> OrderedDict[str, Tensor]:
        consensuated_model = copy.deepcopy(model)
        while self.models_to_consensuate.qsize() > 0:
            weights_and_biases = self.models_to_consensuate.get()
            consensuated_model = Consensus.apply_consensus(
                consensuated_model, weights_and_biases, epsilon=self.epsilon
            )
            self.models_to_consensuate.task_done()
        return consensuated_model

    @staticmethod
    def apply_consensus(
        weights_and_biases_a: OrderedDict[str, Tensor],
        weights_and_biases_b: OrderedDict[str, Tensor],
        epsilon: float = 0.5,
    ) -> OrderedDict[str, Tensor]:
        consensuated_result: OrderedDict[str, Tensor] = OrderedDict()
        for key in weights_and_biases_a.keys():
            if key in weights_and_biases_b:
                consensuated_result[key] = Consensus.consensus_update_to_tensors(
                    tensor_a=weights_and_biases_a[key],
                    tensor_b=weights_and_biases_b[key],
                    epsilon=epsilon,
                )
            else:
                raise ValueError(
                    f"Consensus error. The key '{key}' is not present in both models."
                )
        return consensuated_result

    @staticmethod
    def consensus_update_to_tensors(
        tensor_a: Tensor, tensor_b: Tensor, epsilon: float = 0.5
    ) -> Tensor:
        return epsilon * tensor_a + (1 - epsilon) * tensor_b
