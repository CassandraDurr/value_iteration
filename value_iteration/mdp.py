"""Module containing logic for Markov Decision Processes (MDP) storage, validation and loading."""

from dataclasses import dataclass
from typing import Any

import pandas as pd


def load_mdp_from_csv(
    transitions_filepath: str,
) -> tuple[
    list[str], dict[str, list[str]], dict[tuple[str], float], dict[tuple[str], float]
]:
    """Load the Markov Decision Process (MDP) variables from a CSV file.

    Create variables for states, actions, transition probabilities, and rewards.

    Args:
        transitions_filepath (str): Path to the CSV containing transition information
            ('state', 'action', 'next_state', 'probability', 'reward').

    Raises:
        ValueError: Error in reading in the transitions CSV.
        ValueError: Invalid transition probabilities in the transitions CSV.
        ValueError: Inconsistent rewards for a state-action pair.

    Returns:
        tuple[
            list[str], dict[str, list[str]], dict[tuple[str], float], dict[tuple[str], float]
        ]: states, actions, probabilities, rewards
            states: List of states.
            actions: States and corresponding possible actions.
            probabilities: Transition probabilities {(s, a, s'): prob}.
            rewards: Rewards {(s, a): reward}.
    """
    # Read transitions CSV forcing column names and types
    try:
        transitions_df = pd.read_csv(
            transitions_filepath,
            dtype={"state": str, "action": str, "next_state": str},
            usecols=["state", "action", "next_state", "probability", "reward"],
        )
    except Exception as e:
        raise ValueError(
            f"Error reading transitions CSV: {e}.\n"
            "Ensure it has columns: state, action, next_state, probability, reward."
        ) from e

    # Ensure that the probability and reward columns are numeric
    transitions_df["probability"] = pd.to_numeric(
        transitions_df["probability"], errors="coerce"
    )
    transitions_df["reward"] = pd.to_numeric(transitions_df["reward"], errors="coerce")

    # Validate the transition probabilities
    if (
        transitions_df["probability"].isnull().any()
        or (transitions_df["probability"] < 0).any()
        or (transitions_df["probability"] > 1).any()
    ):
        raise ValueError(
            "Probability column must contain values strictly between 0 and 1."
        )

    # Extract unique states list
    states = list(set(transitions_df["state"]).union(set(transitions_df["next_state"])))

    # Build actions dictionary
    actions = {}
    for _, row in transitions_df.iterrows():
        s, a = row["state"], row["action"]
        if s not in actions:
            actions[s] = []
        if a not in actions[s]:
            actions[s].append(a)

    # Build transition probabilities and rewards dictionaries
    probabilities = {}
    rewards = {}

    # Track unique rewards for (s, a) pairs
    unique_reward_tracker = {}

    for _, row in transitions_df.iterrows():
        s, a, s_ = (
            row["state"],
            row["action"],
            row["next_state"],
        )

        # Store transition probability
        probabilities[(s, a, s_)] = row["probability"]

        # Ensure only one unique reward per (s, a) exists
        reward = row["reward"]
        if (s, a) in unique_reward_tracker and unique_reward_tracker[(s, a)] != reward:
            raise ValueError(
                f"Inconsistent rewards for state-action pair ({s}, {a}): "
                f"Found both {unique_reward_tracker[(s, a)]} and {reward}."
            )

        # Store the reward
        unique_reward_tracker[(s, a)] = reward
        rewards[(s, a)] = reward

    return states, actions, probabilities, rewards


@dataclass
class MDP:
    """Data class for storing and validating Markov Decision Process (MDP) variables."""

    states: list
    actions: dict[Any, list[Any]]
    probabilities: dict[tuple[Any, Any, Any], float]
    rewards: dict[tuple[Any, Any], float | int]

    def validate(self) -> None:
        """Validate MDP variables.

        Raises:
            ValueError: states must be a list; actions, probabilities, rewards must be dictionaries.
            ValueError: States from actions do not exist in the state list states.
            ValueError: Actions for states in actions must be in a list.
            ValueError: States or next states in probabilities do not exist in states.
            ValueError: Actions in probabilities are not valid for the given state.
            ValueError: Invalid probabilities in probabilities.
            ValueError: States in rewards do not exist in states.
            ValueError: Actions in rewards are not valid for the given state.
            ValueError: Rewards in rewards must be numeric.
        """
        # Validate variable types
        if (
            not isinstance(self.states, list)
            or not isinstance(self.actions, dict)
            or not isinstance(self.probabilities, dict)
            or not isinstance(self.rewards, dict)
        ):
            raise ValueError(
                "states must be a list & actions, probabilities, rewards must be dictionaries."
            )

        # Validate actions
        for state, state_actions in self.actions.items():
            if state not in self.states:
                raise ValueError(f"State {state} in actions does not exist in states.")
            if not isinstance(state_actions, list):
                raise ValueError(f"Actions for state {state} must be in a list.")

        # Validate transition probabilities
        for (s, a, s_), prob in self.probabilities.items():
            if s not in self.states or s_ not in self.states:
                raise ValueError(
                    f"State {s} or {s_} in probabilities is not in states."
                )
            if s in self.actions and a not in self.actions[s]:
                raise ValueError(
                    f"Action {a} in probabilities is not a valid action for state {s}."
                )
            if not isinstance(prob, (int, float)) or prob < 0 or prob > 1:
                raise ValueError(
                    f"Probability for ({s}, {a}, {s_}) must be between 0 and 1."
                )

        # Validate rewards
        for (s, a), reward in self.rewards.items():
            if s not in self.states:
                raise ValueError(f"State {s} in rewards is not in states.")
            if s in self.actions and a not in self.actions[s]:
                raise ValueError(
                    f"Action {a} in rewards is not a valid action for state {s}."
                )
            if not isinstance(reward, (int, float)):
                raise ValueError(f"Reward for ({s}, {a}) must be a number.")

    def __post_init__(self):
        """Automatically validate MDP on data class creation."""
        self.validate()
