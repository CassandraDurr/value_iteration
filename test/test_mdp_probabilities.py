"""Tests for validating the probabilities variable from the MDP data class."""

import pytest

from value_iteration import MDP

valid_S = ["S1", "S3", "S2"]
valid_A = {"S1": ["A1"], "S2": ["A2"]}
valid_R = {("S1", "A1"): 5, ("S2", "A2"): 10}


@pytest.mark.parametrize(
    "example_input,expectation",
    [
        (
            {
                ("S1", "A1", "S2"): 0.8,
                ("S1", "A1", "S3"): 0.2,
                ("S2", "A2", "S1"): 1.0,
                ("S3", "A2", "Random State"): 1.0,
            },
            pytest.raises(
                ValueError,
                match="in probabilities is not in states.",
            ),
        ),
        (
            {
                ("S1", "A1", "S2"): 0.8,
                ("S1", "A1", "S3"): 0.2,
                ("S2", "A2", "S1"): 1.0,
                ("Random State", "A2", "S2"): 1.0,
            },
            pytest.raises(
                ValueError,
                match="in probabilities is not in states.",
            ),
        ),
        (
            {
                ("S1", "A1", "S2"): 0.8,
                ("S1", "A1", "S3"): 0.2,
                ("S1", "A2", "S2"): 1.0,
                ("S2", "A2", "S1"): 1.0,
            },
            pytest.raises(
                ValueError,
                match="Invalid state-action pair",
            ),
        ),
        (
            {
                ("S1", "A1", "S2"): 0.8,
                ("S1", "A1", "S3"): 0.2,
                ("S2", "A2", "S1"): 1.1,
            },
            pytest.raises(
                ValueError,
                match="must be between 0 and 1.",
            ),
        ),
        (
            {
                ("S1", "A1", "S2"): 0.8,
                ("S1", "A1", "S3"): 0.2,
                ("S2", "A2", "S1"): -1.1,
            },
            pytest.raises(
                ValueError,
                match="must be between 0 and 1.",
            ),
        ),
        (
            {
                ("S1", "A1", "S2"): 0.8,
                ("S1", "A1", "S3"): 0.2,
                ("S2", "A2", "S1"): "1.0",
            },
            pytest.raises(
                ValueError,
                match="must be between 0 and 1.",
            ),
        ),
        (
            {
                ("S1", "A1", "S2"): 0.6,
                ("S1", "A1", "S3"): 0.2,
                ("S2", "A2", "S1"): 1.0,
            },
            pytest.raises(
                ValueError,
                match="must sum to 1, found",
            ),
        ),
    ],
)
def test_validate_probabilities(example_input, expectation):
    """Test the validation of probabilities variable from MDP data class."""
    with expectation:
        MDP(
            states=valid_S,
            actions=valid_A,
            probabilities=example_input,
            rewards=valid_R,
        )
