import copy

import torch

from macofl.datatypes.consensus import Consensus


def test_consensus_update_tensors():
    # Two input tensors of 3x3: one filled with zeros, the other filled with tens
    tensor_zeros = torch.zeros((3, 3))
    tensor_tens = torch.full((3, 3), 10.0)

    # Expected output after applying consensus
    expected_tensor = torch.full((3, 3), 5.0)

    # Apply the consensus update
    consensuated_tensor = Consensus.consensus_update_to_tensors(
        tensor_zeros, tensor_tens, epsilon=0.5
    )

    # Check that the output is a tensor of fives
    assert torch.allclose(
        consensuated_tensor, expected_tensor
    ), f"Expected tensor of 5s but got {consensuated_tensor}"


def test_consensus_update_models():
    consensus = Consensus(
        epsilon=0.5, max_order=2, max_seconds_to_accept_pre_consensus=10
    )
    # Define the state dictionaries of two models with tensors of zeros and tens
    model_state_a = {"weight": torch.zeros((3, 3)), "bias": torch.zeros((3,))}
    model_state_b = {"weight": torch.full((3, 3), 10.0), "bias": torch.full((3,), 10.0)}

    freeze_model_a = copy.deepcopy(model_state_a)

    # Expected output after applying consensus
    expected_model_state = {
        "weight": torch.full((3, 3), 5.0),
        "bias": torch.full((3,), 5.0),
    }

    # Apply the consensus algorithm
    consensuated_state_dict = consensus.apply_consensus(
        model_state_a, model_state_b, epsilon=0.5
    )

    # Check that both 'weight' and 'bias' are correct
    assert torch.allclose(
        consensuated_state_dict["weight"], expected_model_state["weight"]
    ), f"Expected weight tensor of 5s but got {consensuated_state_dict['weight']}"
    assert torch.allclose(
        consensuated_state_dict["bias"], expected_model_state["bias"]
    ), f"Expected bias tensor of 5s but got {consensuated_state_dict['bias']}"

    # Check that initial model is not modified
    assert torch.allclose(
        freeze_model_a["weight"], model_state_a["weight"]
    ), "The initial model has been modified during consensus process"


def test_apply_consensus_algorithm():
    consensus = Consensus(
        epsilon=0.5, max_order=2, max_seconds_to_accept_pre_consensus=10
    )

    # Define the models
    agent_model = {"weight": torch.zeros((3, 3)), "bias": torch.zeros((3,))}
    neighbour_models = [
        {"weight": torch.full((3, 3), 10.0), "bias": torch.full((3,), 10.0)},
        {"weight": torch.full((3, 3), 7.0), "bias": torch.full((3,), 7.0)},
        {"weight": torch.full((3, 3), 3.0), "bias": torch.full((3,), 3.0)},
    ]

    # Expected output after applying consensus
    expected_model_state = {
        "weight": torch.full((3, 3), 4.5),
        "bias": torch.full((3,), 4.5),
    }

    # Apply consensus
    for weights_and_biases in neighbour_models:
        consensus.models_to_consensuate.put(weights_and_biases)
    agent_model = consensus.apply_all_consensus(agent_model)

    # Check that both 'weight' and 'bias' are correct
    assert torch.allclose(
        agent_model["weight"], expected_model_state["weight"]
    ), f"Expected weight tensor of 5s but got {agent_model['weight']}"
    assert torch.allclose(
        agent_model["bias"], expected_model_state["bias"]
    ), f"Expected bias tensor of 5s but got {agent_model['bias']}"
