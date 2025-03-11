"""Tests for function load_mdp_from_csv."""

import os

import pytest

from value_iteration import load_mdp_from_csv

# Get the path of the test data folder
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def test_load_valid_mdp():
    """Test a valid MDP configuration returns expected values."""
    testing_csv = os.path.join(TEST_DATA_DIR, "valid_mdp.csv")
    states, actions, probabilities, rewards = load_mdp_from_csv(testing_csv)

    assert set(states) == {"S1", "S2", "S3"}
    assert actions == {"S1": ["A1"], "S2": ["A2"]}
    assert probabilities[("S1", "A1", "S2")] == 0.8
    assert probabilities[("S1", "A1", "S3")] == 0.2
    assert probabilities[("S2", "A2", "S1")] == 1.0
    assert rewards[("S1", "A1")] == 5
    assert rewards[("S2", "A2")] == 10


def test_load_invalid_column_names():
    """Test a MDP configuration which has invalid column names."""
    testing_csv = os.path.join(TEST_DATA_DIR, "incorrect_column_name.csv")
    with pytest.raises(ValueError, match="Ensure it has columns"):
        load_mdp_from_csv(testing_csv)


def test_load_missing_columns():
    """Test a MDP configuration with a missing column."""
    testing_csv = os.path.join(TEST_DATA_DIR, "missing_columns.csv")
    with pytest.raises(ValueError, match="Ensure it has columns"):
        load_mdp_from_csv(testing_csv)


def test_invalid_probability_values():
    """Test a MDP configuration with an invalid, negative probability."""
    testing_csv = os.path.join(TEST_DATA_DIR, "invalid_probability.csv")
    with pytest.raises(
        ValueError,
        match="Probability column must contain values strictly between 0 and 1",
    ):
        load_mdp_from_csv(testing_csv)


def test_inconsistent_rewards():
    """Test a MDP configuration where a state-action pair has inconsistent rewards."""
    testing_csv = os.path.join(TEST_DATA_DIR, "inconsistent_rewards.csv")
    with pytest.raises(ValueError, match="Inconsistent rewards for state-action pair"):
        load_mdp_from_csv(testing_csv)
