"""Tests for validating the rewards variable from the MDP data class."""

import pytest

from value_iteration import MDP

valid_S = ["S1", "S3", "S2"]
valid_A = {"S1": ["A1"], "S2": ["A2"]}
valid_P = {("S1", "A1", "S2"): 0.8, ("S1", "A1", "S3"): 0.2, ("S2", "A2", "S1"): 1.0}


@pytest.mark.parametrize(
    "example_input,expectation",
    [
        (
            {
                ("S1", "A1", "S2"): 5,
                ("S1", "A1", "S3"): 5,
                ("S2", "A2", "S1"): 10,
                ("Random state", "A2", "S1"): 10,
            },
            pytest.raises(
                ValueError,
                match="in rewards is not in states.",
            ),
        ),
        (
            {
                ("S1", "A1", "S2"): 5,
                ("S1", "A1", "S3"): 5,
                ("S2", "A2", "S1"): 10,
                ("S3", "A2", "Random state"): 10,
            },
            pytest.raises(
                ValueError,
                match="in rewards is not in states.",
            ),
        ),
        (
            {
                ("S1", "A1", "S2"): 5,
                ("S1", "A1", "S3"): 5,
                ("S2", "A2", "S1"): 10,
                ("S1", "A2", "S2"): 4,
            },
            pytest.raises(
                ValueError,
                match="Invalid state-action pair",
            ),
        ),
        (
            {
                ("S1", "A1", "S2"): 5,
                ("S1", "A1", "S3"): 5,
                ("S2", "A2", "S1"): "10",
            },
            pytest.raises(
                ValueError,
                match="must be a number",
            ),
        ),
    ],
)
def test_validate_rewards(example_input, expectation):
    """Test the validation of rewards variable from MDP data class."""
    with expectation:
        MDP(
            states=valid_S,
            actions=valid_A,
            probabilities=valid_P,
            rewards=example_input,
        )
