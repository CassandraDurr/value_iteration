"""Module for value iteration functions and validation."""

from typing import Any

import pandas as pd


def load_mdp_from_csv(
    transitions_filepath: str, state_actions_filepath: str
) -> tuple[
    list[str], dict[str, list[str]], dict[tuple[str], float], dict[tuple[str], float]
]:
    """Load the Markov Decision Process (MDP) variables from CSV files.

    Create variables for states, actions, transition probabilities, and rewards.

    Args:
        transitions_filepath (str): Path to CSV file containing transitions
            ('state', 'action', 'next_state', 'probability', 'reward').
        state_actions_filepath (str): Path to CSV file containing state-actions space
            ('state', 'actions').

    Raises:
        ValueError: Error in reading in transitions CSV.
        ValueError: Invalid transition probabilities in transitions CSV.
        ValueError: Error in reading in state-actions CSV.

    Returns:
        tuple[
            list[str], dict[str, list[str]], dict[tuple[str], float], dict[tuple[str], float]
        ]: states, actions, probabilities, rewards
            states: List of states.
            actions: States and corresponding possible actions.
            probabilities: Transition probabilities {(s, a, s'): prob}.
            rewards: Rewards {(s, a, s'): reward}.
    """
    # Read transitions CSV - enforce columns and types
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

    # Ensure that probability and reward columns are numeric
    transitions_df["probability"] = pd.to_numeric(
        transitions_df["probability"], errors="coerce"
    )
    transitions_df["reward"] = pd.to_numeric(transitions_df["reward"], errors="coerce")

    # Validate transition probabilities
    if (
        transitions_df["probability"].isnull().any()
        or (transitions_df["probability"] < 0).any()
        or (transitions_df["probability"] > 1).any()
    ):
        raise ValueError(
            "Probability column must contain values strictly between 0 and 1."
        )

    # Read state-actions CSV - enforce columns and types
    try:
        state_actions_df = pd.read_csv(
            state_actions_filepath, dtype=str, usecols=["state", "actions"]
        )
    except Exception as e:
        raise ValueError(
            f"Error reading state-actions CSV: {e}. Ensure it has columns: state, action."
        ) from e

    # Extract unique states list
    states = list(state_actions_df["state"].unique())

    # Build actions dictionary
    actions = {}
    for _, row in state_actions_df.iterrows():
        # Actions should be separated by semicolons
        actions[row["state"]] = row["actions"].split(";")

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

        # Store reward
        rewards[(s, a, s_)] = row["reward"]

    return states, actions, probabilities, rewards


def validate_mdp(
    states: list,
    actions: dict[Any, list[Any]],
    probabilities: dict[tuple[Any], float],
    rewards: dict[tuple[Any], float | int],
) -> None:
    """Validate Markov Decision Process (MDP) variables.

    Args:
        states (list): List of states in the state space.
        actions (dict[Any, list[Any]]): Map of states to their valid actions.
        probabilities (dict[tuple[Any], float]): Map of (s, a, s') to the transition probability.
        rewards (dict[tuple[Any], float | int]): Map of (s, a, s') to the reward.

    Raises:
        ValueError: states must be a list & actions, probabilities, rewards must be dictionaries.
        ValueError: States from actions do not exist in the state list states.
        ValueError: Actions for states in actions must be in a list.
        ValueError: States or next states in probabilities do not exist in states.
        ValueError: Actions in probabilities are not valid for the given state.
        ValueError: Invalid probabilities in probabilities.
        ValueError: States or next states in rewards do not exist in states.
        ValueError: Actions in rewards are not valid for the given state.
        ValueError: Rewards in rewards must be numeric.
    """
    # Validates MDP variable types
    if (
        not isinstance(states, list)
        or not isinstance(actions, dict)
        or not isinstance(probabilities, dict)
        or not isinstance(rewards, dict)
    ):
        raise ValueError(
            "states must be a list & actions, probabilities, rewards must be dictionaries."
        )

    for state, state_actions in actions.items():
        if state not in states:
            raise ValueError(f"State {state} in actions does not exist in states.")
        if not isinstance(state_actions, list):
            raise ValueError(f"Actions for state {state} must be in a list.")

    for (s, a, s_), prob in probabilities.items():
        if s not in states or s_ not in states:
            raise ValueError(f"State {s} or {s_} in probabilities is not in states.")
        if s in actions and a not in actions[s]:
            raise ValueError(
                f"Action {a} in probabilities is not a valid action for state {s}."
            )
        if not isinstance(prob, (int, float)) or prob < 0 or prob > 1:
            raise ValueError(
                f"Probability for ({s}, {a}, {s_}) must be between 0 and 1."
            )

    for (s, a, s_), reward in rewards.items():
        if s not in states or s_ not in states:
            raise ValueError(f"State {s} or {s_} in rewards is not in states.")
        if s in actions and a not in actions[s]:
            raise ValueError(
                f"Action {a} in rewards is not a valid action for state {s}."
            )
        if not isinstance(reward, (int, float)):
            raise ValueError(f"Reward for ({s}, {a}, {s_}) must be a number.")


def value_iteration(
    states: list,
    actions: dict[Any, list[Any]],
    probabilities: dict[tuple[Any], float],
    rewards: dict[tuple[Any], float | int],
    gamma: float = 0.9,
    theta: float = 1e-6,
    printing: bool = False,
) -> tuple[dict[Any, float], dict]:
    """Perform value iteration and strict input validation.

    Args:
        states (list): List of states in the state space.
        actions (dict[Any, list[Any]]): Map of states to their valid actions.
        probabilities (dict[tuple[Any], float]): Map of (s, a, s') to the transition probability.
        rewards (dict[tuple[Any], float  |  int]): Map of (s, a, s') to the reward.
        gamma (float, optional): Discount factor. Defaults to 0.9.
        theta (float, optional): Convergence threshold. Defaults to 1e-6.
        printing (bool, optional): If True, prints out iteration updates. Defaults to False.

    Raises:
        ValueError: Discount factor gamma must be between 0 and 1.
        ValueError: Convergence threshold theta must not be negative.

    Returns:
        tuple[dict[Any, float], dict]: Optimal values, Optimal policy
            Optimal values: Mapping of states to the optimal state value
            Optimal policy: Mapping of states to the best action to take in that state
    """
    # Validate MDP variables
    validate_mdp(states, actions, probabilities, rewards)

    # Validate additional parameters
    if gamma > 1 or gamma < 0:
        raise ValueError("Discount factor gamma must be between 0 and 1.")

    if theta <= 0:
        raise ValueError("Convergence threshold theta must not be negative.")

    # Initialise value function
    values = dict.fromkeys(states, 0.0)

    # Track iterations
    iteration = 0
    while True:
        delta = 0
        iteration += 1
        new_values = values.copy()

        for s in states:
            # If no valid actions exist for this state, pass
            if s not in actions or not actions[s]:
                continue

            old_value = values[s]
            best_value = float("-inf")

            for a in actions[s]:
                q_value = sum(
                    probabilities.get((s, a, s_), 0)
                    * (rewards.get((s, a, s_), 0) + gamma * values[s_])
                    for s_ in states
                )
                best_value = max(best_value, q_value)

            new_values[s] = best_value
            delta = max(delta, abs(old_value - new_values[s]))

        values = new_values

        if printing:
            print(f"Iteration {iteration}, max value change: {delta}")

        if delta < theta:
            break

    # Extract policy
    policy = {}
    for s in states:
        if s not in actions or not actions[s]:
            policy[s] = None
            continue

        best_action = None
        best_value = float("-inf")

        for a in actions[s]:
            q_value = sum(
                probabilities.get((s, a, s_), 0)
                * (rewards.get((s, a, s_), 0) + gamma * values[s_])
                for s_ in states
            )
            if q_value > best_value:
                best_value = q_value
                best_action = a

        policy[s] = best_action

    return values, policy
