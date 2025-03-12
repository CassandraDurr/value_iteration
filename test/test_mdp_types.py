"""Tests for validating the types associated with the MDP data class."""

import pytest

from value_iteration import MDP

valid_S = ["S1", "S3", "S2"]
valid_A = {"S1": ["A1"], "S2": ["A2"]}
valid_P = {("S1", "A1", "S2"): 0.8, ("S1", "A1", "S3"): 0.2, ("S2", "A2", "S1"): 1.0}
valid_R = {("S1", "A1"): 5, ("S2", "A2"): 10}


@pytest.mark.parametrize(
    "example_input,expectation",
    [
        (
            tuple(valid_S),
            pytest.raises(
                ValueError,
                match="states must be a list & actions, probabilities, rewards must be",
            ),
        ),
        (
            set(valid_S),
            pytest.raises(
                ValueError,
                match="states must be a list & actions, probabilities, rewards must be",
            ),
        ),
    ],
)
def test_validate_state_type(example_input, expectation):
    """Test the validation of states type."""
    with expectation:
        MDP(
            states=example_input,
            actions=valid_A,
            probabilities=valid_P,
            rewards=valid_R,
        )


@pytest.mark.parametrize(
    "example_input,expectation",
    [
        (
            tuple(valid_A),
            pytest.raises(
                ValueError,
                match="states must be a list & actions, probabilities, rewards must be",
            ),
        ),
        (
            set(valid_A),
            pytest.raises(
                ValueError,
                match="states must be a list & actions, probabilities, rewards must be",
            ),
        ),
        (
            list(valid_A),
            pytest.raises(
                ValueError,
                match="states must be a list & actions, probabilities, rewards must be",
            ),
        ),
        (
            valid_A.items(),
            pytest.raises(
                ValueError,
                match="states must be a list & actions, probabilities, rewards must be",
            ),
        ),
    ],
)
def test_validate_action_type(example_input, expectation):
    """Test the validation of actions type."""
    with expectation:
        MDP(
            states=valid_S,
            actions=example_input,
            probabilities=valid_P,
            rewards=valid_R,
        )


@pytest.mark.parametrize(
    "example_input,expectation",
    [
        (
            tuple(valid_P),
            pytest.raises(
                ValueError,
                match="states must be a list & actions, probabilities, rewards must be",
            ),
        ),
        (
            set(valid_P),
            pytest.raises(
                ValueError,
                match="states must be a list & actions, probabilities, rewards must be",
            ),
        ),
        (
            list(valid_P),
            pytest.raises(
                ValueError,
                match="states must be a list & actions, probabilities, rewards must be",
            ),
        ),
        (
            valid_P.items(),
            pytest.raises(
                ValueError,
                match="states must be a list & actions, probabilities, rewards must be",
            ),
        ),
    ],
)
def test_validate_probabilities_type(example_input, expectation):
    """Test the validation of probabilities type."""
    with expectation:
        MDP(
            states=valid_S,
            actions=valid_A,
            probabilities=example_input,
            rewards=valid_R,
        )


@pytest.mark.parametrize(
    "example_input,expectation",
    [
        (
            tuple(valid_R),
            pytest.raises(
                ValueError,
                match="states must be a list & actions, probabilities, rewards must be",
            ),
        ),
        (
            set(valid_R),
            pytest.raises(
                ValueError,
                match="states must be a list & actions, probabilities, rewards must be",
            ),
        ),
        (
            list(valid_R),
            pytest.raises(
                ValueError,
                match="states must be a list & actions, probabilities, rewards must be",
            ),
        ),
        (
            valid_R.items(),
            pytest.raises(
                ValueError,
                match="states must be a list & actions, probabilities, rewards must be",
            ),
        ),
    ],
)
def test_validate_rewards_type(example_input, expectation):
    """Test the validation of rewards type."""
    with expectation:
        MDP(
            states=valid_S,
            actions=valid_A,
            probabilities=valid_P,
            rewards=example_input,
        )
