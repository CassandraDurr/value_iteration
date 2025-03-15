"""Tests for validating the ValueIteration and AsynchValueIteration classes."""

import pytest

from value_iteration import MDP, AsynchValueIteration, ValueIteration

valid_S = ["S1", "S3", "S2"]
valid_A = {"S1": ["A1"], "S2": ["A2"]}
valid_P = {("S1", "A1", "S2"): 0.8, ("S1", "A1", "S3"): 0.2, ("S2", "A2", "S1"): 1.0}
valid_R = {("S1", "A1", "S2"): 5, ("S1", "A1", "S3"): 5, ("S2", "A2", "S1"): 10}
valid_mdp = MDP(states=valid_S, actions=valid_A, probabilities=valid_P, rewards=valid_R)


@pytest.mark.parametrize(
    "example_input,expectation",
    [
        (
            1.5,
            pytest.raises(
                ValueError,
                match="Discount factor gamma must be between 0 and 1.",
            ),
        ),
        (
            -0.7,
            pytest.raises(
                ValueError,
                match="Discount factor gamma must be between 0 and 1.",
            ),
        ),
    ],
)
def test_validate_value_iteration_gamma(example_input, expectation):
    """Test the validation of gamma variable for ValueIteration class."""
    with expectation:
        ValueIteration(mdp=valid_mdp, gamma=example_input)


@pytest.mark.parametrize(
    "example_input,expectation",
    [
        (
            1.5,
            pytest.raises(
                ValueError,
                match="Discount factor gamma must be between 0 and 1.",
            ),
        ),
        (
            -0.7,
            pytest.raises(
                ValueError,
                match="Discount factor gamma must be between 0 and 1.",
            ),
        ),
    ],
)
def test_validate_async_value_iteration_gamma(example_input, expectation):
    """Test the validation of gamma variable for AsynchValueIteration class."""
    with expectation:
        AsynchValueIteration(mdp=valid_mdp, gamma=example_input)


@pytest.mark.parametrize(
    "example_input,expectation",
    [
        (
            ValueIteration,
            pytest.raises(
                ValueError,
                match="Convergence threshold theta must not be negative.",
            ),
        ),
        (
            AsynchValueIteration,
            pytest.raises(
                ValueError,
                match="Convergence threshold theta must not be negative.",
            ),
        ),
    ],
)
def test_validate_theta(example_input, expectation):
    """Test the validation of theta variable for value iteration classes."""
    with expectation:
        example_input(mdp=valid_mdp, theta=-0.7)


def test_value_itr_init_helper():
    """Test value iteration initialisation and helper functions."""
    value_itr = ValueIteration(mdp=valid_mdp)
    async_value_itr = AsynchValueIteration(mdp=valid_mdp)

    # Initialised values
    assert value_itr.values == {"S1": 0, "S3": 0, "S2": 0}
    assert async_value_itr.valid_state_actions == {("S1", "A1"), ("S2", "A2")}
    assert async_value_itr.q_table == {("S1", "A1"): 0, ("S2", "A2"): 0}

    # Get q value
    assert value_itr.get_q_value("S2", "A1") == 0
    assert async_value_itr.get_q_value("S2", "A1") == 0
    assert async_value_itr.get_q_value("S2", "A2") == 0

    # Valid state-action pair
    async_value_itr.update_q_table("S2", "A2")
    assert async_value_itr.q_table[("S2", "A2")] > 0

    # Invalid state-action pair
    async_value_itr.update_q_table("S3", "A1")
    assert async_value_itr.q_table[("S3", "A1")] == 0

    # Check extract policy
    async_value_itr.q_table = {("S1", "A1"): 10, ("S2", "A2"): 5}
    assert async_value_itr.extract_policy() == {"S1": "A1", "S2": "A2", "S3": None}


def test_value_itr_method():
    """Test value iteration functions for sync and asyc value iteration."""
    value_itr = ValueIteration(mdp=valid_mdp)
    _, optimal_policy = value_itr.value_iteration()

    assert optimal_policy == {"S1": "A1", "S2": "A2", "S3": None}

    async_value_itr = AsynchValueIteration(mdp=valid_mdp)
    _, optimal_async_policy = async_value_itr.value_iteration()

    assert optimal_async_policy == {"S1": "A1", "S2": "A2", "S3": None}
