import copy

import torch

from macofl.datatypes.consensus_manager import ConsensusManager


def test_consensus_update_tensors():
    max_order = 2
    epsilon_margin = 0.05

    # Calculate epsilon to know the expected result
    epsilon = (1 / max_order) - epsilon_margin

    # Two input tensors of 3x3: one filled with zeros, the other filled with tens
    tensor_zeros = torch.zeros((3, 3))
    tensor_tens = torch.full((3, 3), 10.0)

    # Expected output after applying consensus
    expected_tensor = torch.full((3, 3), (1 - epsilon) * 10)

    # Apply the consensus update
    consensuated_tensor = ConsensusManager.apply_consensus_to_tensors(
        tensor_zeros, tensor_tens, max_order=max_order, epsilon_margin=epsilon_margin
    )

    # Check that the output is a tensor of fives
    assert torch.allclose(
        consensuated_tensor, expected_tensor
    ), f"Expected tensor of 5s but got {consensuated_tensor}"


def test_consensus_update_models():
    max_order = 2
    epsilon_margin = 0.05

    # Calculate epsilon to know the expected result
    epsilon = (1 / max_order) - epsilon_margin

    # Define the state dictionaries of two models with tensors of zeros and tens
    model_state_a = {"weight": torch.zeros((3, 3)), "bias": torch.zeros((3,))}
    model_state_b = {"weight": torch.full((3, 3), 10.0), "bias": torch.full((3,), 10.0)}

    freeze_model_a = copy.deepcopy(model_state_a)

    # Expected output after applying consensus
    expected_model_state = {
        "weight": torch.full((3, 3), 10.0 * (1 - epsilon)),
        "bias": torch.full((3,), 10.0 * (1 - epsilon)),
    }

    # Apply the consensus algorithm
    consensuated_state_dict = ConsensusManager.apply_consensus_to_layers(
        model_state_a, model_state_b, max_order=max_order, epsilon_margin=epsilon_margin
    )

    # Check that both 'weight' and 'bias' are correct
    assert torch.allclose(
        consensuated_state_dict["weight"],
        expected_model_state["weight"],
    ), f"Expected weight tensor of 5s but got {consensuated_state_dict['weight']}"
    assert torch.allclose(
        consensuated_state_dict["bias"],
        expected_model_state["bias"],
    ), f"Expected bias tensor of 5s but got {consensuated_state_dict['bias']}"

    # Check that initial model is not modified
    assert torch.allclose(
        freeze_model_a["weight"], model_state_a["weight"]
    ), "The initial model has been modified during consensus process"


def test_consensus_update_layers():
    max_order = 2
    epsilon_margin = 0.05

    # Calculate epsilon to know the expected result
    epsilon = (1 / max_order) - epsilon_margin

    # Define the state dictionaries of two models with tensors of zeros and tens
    full_model = {"weight": torch.zeros((3, 3)), "bias": torch.zeros((3,))}
    layers = {"bias": torch.full((3,), 10.0)}

    freeze_model = copy.deepcopy(full_model)

    # Expected output after applying consensus
    expected_model_state = {
        "weight": torch.zeros((3, 3)),
        "bias": torch.full((3,), 10.0 * (1 - epsilon)),
    }

    # Apply the consensus algorithm
    consensuated_state_dict = ConsensusManager.apply_consensus_to_layers(
        full_model=full_model,
        layers=layers,
        max_order=max_order,
        epsilon_margin=epsilon_margin,
    )

    # Check that both 'weight' and 'bias' are correct
    assert torch.allclose(
        consensuated_state_dict["weight"],
        expected_model_state["weight"],
    ), f"Expected weight tensor of 5s but got {consensuated_state_dict['weight']}"
    assert torch.allclose(
        consensuated_state_dict["bias"],
        expected_model_state["bias"],
    ), f"Expected bias tensor of 5s but got {consensuated_state_dict['bias']}"

    # Check that initial model is not modified
    assert torch.allclose(
        freeze_model["weight"], full_model["weight"]
    ), "The initial model has been modified during consensus process"
