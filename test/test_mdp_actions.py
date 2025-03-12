"""Tests for validating the actions variable from the MDP data class."""

import pytest

from value_iteration import MDP

valid_S = ["S1", "S3", "S2"]
valid_P = {("S1", "A1", "S2"): 0.8, ("S1", "A1", "S3"): 0.2, ("S2", "A2", "S1"): 1.0}
valid_R = {("S1", "A1"): 5, ("S2", "A2"): 10}


@pytest.mark.parametrize(
    "example_input,expectation",
    [
        (
            {"S1": ["A1"], "S2": ["A2"], "Random State": ["A1", "A2"]},
            pytest.raises(
                ValueError,
                match="Some states in actions are not in states.",
            ),
        ),
        (
            {"S1": ("A1"), "S2": ["A2"], "S3": ["A1", "A2"]},
            pytest.raises(
                ValueError,
                match="Actions for each state must be stored as a list.",
            ),
        ),
        (
            {"S1": {"A1"}, "S2": ["A2"], "S3": ["A1", "A2"]},
            pytest.raises(
                ValueError,
                match="Actions for each state must be stored as a list.",
            ),
        ),
    ],
)
def test_validate_actions(example_input, expectation):
    """Test the validation of actions variable from MDP data class."""
    with expectation:
        MDP(
            states=valid_S,
            actions=example_input,
            probabilities=valid_P,
            rewards=valid_R,
        )
