import copy
from queue import Queue
from typing import OrderedDict

import numpy as np
import torch
from torch import Tensor


class Consensus:

    def __init__(
        self,
        epsilon: int,
        iterations: int,
        max_seconds_to_accept_pre_consensus: float,
    ) -> None:
        self.weights: Queue[Tensor] = Queue()
        self.biases: Queue[Tensor] = Queue()
        self.epsilon = epsilon
        if epsilon < 0 or epsilon > 1:
            raise RuntimeError("Consensus epsilon must be in [0, 1] range.")
        self.consensus_iterations: int = iterations
        self.max_seconds_to_accept_pre_consensus: float = (
            max_seconds_to_accept_pre_consensus
        )

    def apply_consensus(self, own_weights, neighbour_weights, epsilon) -> Tensor:
        """
        Apply the asynchronous consensus between the weights of the agent and those of its neighbour
        :param own_weights: weights of the agent
        :param neighbour_weights: weights of the neighbour
        :param eps: epsilon value
        :return: the new weights post-consensus
        """
        average_weights = copy.deepcopy(own_weights)
        for key in own_weights[0].keys():
            if len(own_weights[0][key]) != len(neighbour_weights[0][key]):
                raise RuntimeError(
                    "Consensus can only be applied to arrays of same length"
                )

            temp_own = own_weights[0][key].numpy()
            temp_neighbour = neighbour_weights[0][key].numpy()

            temp_own_flat = temp_own.flatten()
            temp_neighbour_flat = temp_neighbour.flatten()

            temp_consensus = temp_own_flat + epsilon * (
                temp_neighbour_flat - temp_own_flat
            )
            average_weights[0][key] = torch.from_numpy(
                np.reshape(temp_consensus, temp_own.shape)
            )
        return average_weights
