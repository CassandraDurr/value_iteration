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

    Returns:
        tuple[
            list[str], dict[str, list[str]], dict[tuple[str], float], dict[tuple[str], float]
        ]: states, actions, probabilities, rewards
            states: List of states.
            actions: States and corresponding possible actions.
            probabilities: Transition probabilities {(s, a, s'): prob}.
            rewards: Rewards {(s, a, s'): reward}.
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

    for _, row in transitions_df.iterrows():
        s, a, s_ = (
            row["state"],
            row["action"],
            row["next_state"],
        )

        # Store transition probability
        probabilities[(s, a, s_)] = row["probability"]

        # Store the reward
        rewards[(s, a, s_)] = row["reward"]

    return states, actions, probabilities, rewards


@dataclass
class MDP:
    """Data class for storing and validating Markov Decision Process (MDP) variables."""

    states: list
    actions: dict[Any, list[Any]]
    probabilities: dict[tuple[Any, Any, Any], float]
    rewards: dict[tuple[Any, Any, Any], float | int]

    def validate_types(self) -> None:
        """Validate that the types of each MDP variable is as expected.

        Raises:
            ValueError: States should be lists, the rest should be dictionaries.
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

    def validate_actions(self) -> None:
        """Validate that the actions dictionary is as expected.

        Raises:
            ValueError: Some states exists in actions which are invalid.
            ValueError: The actions per state should be stored as a list.
        """
        # Ensure every state in actions exists in states
        if any(state not in self.states for state in self.actions):
            raise ValueError("Some states in actions are not in states.")

        # Ensure actions are lists
        if any(
            not isinstance(action_list, list) for action_list in self.actions.values()
        ):
            raise ValueError("Actions for each state must be stored as a list.")

    def validate_probabilities(self) -> None:
        """Validate the transitions probability dictionary is as expected.

        Raises:
            ValueError: The current or next state is invalid.
            ValueError: An invalid (s,a) pair has non-zero probability.
            ValueError: Non-numeric probability or probability not in range [0,1].
            ValueError: Probability for valid state-action pair doesn't sum to 1.
        """
        # Get valid state-action pairs to check if probabilies sum to 1
        valid_state_actions = {
            (state, action)
            for state, actions in self.actions.items()
            for action in actions
        }

        # Track the sum of probabilities per valid state-action pair
        cumulative_probabilities = dict.fromkeys(valid_state_actions, 0.0)

        for (s, a, s_), prob in self.probabilities.items():
            # Ensure current and next states exist
            if s not in self.states or s_ not in self.states:
                raise ValueError(
                    f"State {s} or {s_} in probabilities is not in states."
                )

            # Ensure (state, action) pairs are valid
            if s in self.actions and a not in self.actions[s]:
                raise ValueError(
                    f"Invalid state-action pair ({s}, {a}) in transition probabilities."
                )

            # Ensure valid probabilities
            if not isinstance(prob, (int, float)) or prob < 0 or prob > 1:
                raise ValueError(
                    f"Probability for ({s}, {a}, {s_}) must be between 0 and 1."
                )

            cumulative_probabilities[(s, a)] += prob

        # Ensure all valid (state, action) probabilities sum to 1
        for (s, a), total_prob in cumulative_probabilities.items():
            if (total_prob <= 1 - 1e-9) or (total_prob >= 1 + 1e-9):
                raise ValueError(
                    f"Total probability for ({s}, {a}) must sum to 1, found {total_prob}."
                )

    def validate_rewards(self) -> None:
        """Validate the rewards dictionary is as expected.

        Raises:
            ValueError: The state or next state in rewards is invalid.
            ValueError: An invalid (s,a) pair has a reward.
            ValueError: The reward is non-numeric.
        """
        # Validate rewards
        for (s, a, s_), reward in self.rewards.items():
            # Ensure states exist
            if s not in self.states or s_ not in self.states:
                raise ValueError(f"State {s} or {s_} in rewards is not in states.")

            # Ensure (state, action) pairs are valid
            if s in self.actions and a not in self.actions[s]:
                raise ValueError(
                    f"Invalid state-action pair ({s}, {a}) in transition probabilities."
                )

            # Ensure numeric rewards
            if not isinstance(reward, (int, float)):
                raise ValueError(f"Reward for ({s}, {a}) must be a number.")

    def __post_init__(self):
        """Automatically validate MDP on data class creation."""
        self.validate_types()
        self.validate_actions()
        self.validate_probabilities()
        self.validate_rewards()
