"""Module for value iteration."""

import random
from typing import Any

from .mdp import MDP


def extract_policy_from_values(
    mdp: MDP,
    values: dict[Any, float],
    gamma: float = 0.9,
) -> dict:
    """Given Markov Decision Process (MDP) variables and value function, return the optimal policy.

    Args:
        mdp (MDP): Data class containing states, actions, rewards and probabilities.
        values (dict[Any, float]): Value function of states.
        gamma (float, optional): Discount factor. Defaults to 0.9.

    Returns:
        dict: Optimal policy - mapping of states to best action.
    """
    # Extract policy
    policy = {}
    for s in mdp.states:
        # Initialise best action and value for state
        best_action = None
        best_value = float("-inf")

        for a in mdp.actions[s]:
            q_value = mdp.rewards.get((s, a), 0) + gamma * sum(
                mdp.probabilities.get((s, a, s_), 0) * values[s_] for s_ in mdp.states
            )

            if q_value > best_value:
                best_value = q_value
                best_action = a

        policy[s] = best_action

    return policy


def value_iteration(
    mdp: MDP,
    gamma: float = 0.9,
    theta: float = 1e-6,
    printing: bool = False,
) -> tuple[dict[Any, float], dict]:
    """Perform value iteration and strict input validation.

    Args:
        mdp (MDP): Data class containing states, actions, rewards and probabilities.
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
    # Validate value iteration parameters
    if gamma > 1 or gamma < 0:
        raise ValueError("Discount factor gamma must be between 0 and 1.")

    if theta <= 0:
        raise ValueError("Convergence threshold theta must not be negative.")

    # Initialise value function as zero for all states
    values = dict.fromkeys(mdp.states, 0.0)

    # Track iterations
    iteration = 0
    while True:
        iteration += 1

        # Copy existing value function
        new_values = values.copy()

        # Track changes in the value function for convergence
        delta_values = []

        for s in mdp.states:
            # If no valid actions exist for this state, pass
            if s not in mdp.actions or not mdp.actions[s]:
                continue

            # Initialise list to store Q values for the state
            q_values = []

            # Get Q for all actions for this state
            for a in mdp.actions[s]:
                q_values.append(
                    mdp.rewards.get((s, a), 0)
                    + gamma
                    * sum(
                        mdp.probabilities.get((s, a, s_), 0) * values[s_]
                        for s_ in mdp.states
                    )
                )

            # Update value function for the state
            new_values[s] = max(q_values)

            # Calculate absolute change in value function for the state
            delta_values.append(abs(values[s] - new_values[s]))

        # Update value function
        values = new_values

        if printing:
            print(f"Iteration {iteration}, max value change: {max(delta_values)}")

        # Test for convergence
        if max(delta_values) < theta:
            break

    # Extract policy
    policy = extract_policy_from_values(mdp, values, gamma)

    return values, policy


def extract_policy_from_q_table(
    mdp: MDP, q_table: dict[tuple[Any, Any], float]
) -> dict:
    """Given Markov Decision Process (MDP) variables and Q-values, return the optimal policy.

    Args:
        mdp (MDP): Data class containing states, actions, rewards and probabilities.
        q_table (dict[tuple[Any, Any], float]): Q-values associated with state-action pairs.

    Returns:
        dict: Optimal policy - mapping of states to best action.
    """
    policy = {}
    for state in mdp.states:
        # Get all actions for the current state
        actions = mdp.actions.get(state, [])

        # Choose the action with the maximum Q-value
        best_action = None
        max_q_value = float("-inf")

        for action in actions:
            q_value = q_table.get((state, action), 0)
            if q_value > max_q_value:
                max_q_value = q_value
                best_action = action

        # Assign the best action for the state
        policy[state] = best_action

    return policy


def asynchronous_value_iteration(
    mdp: MDP,
    gamma: float = 0.9,
    theta: float = 1e-6,
    printing: bool = False,
    n: int = 10,
) -> tuple[dict[Any, float], dict]:
    """Perform value iteration and strict input validation.

    Args:
        mdp (MDP): Data class containing states, actions, rewards and probabilities.
        gamma (float, optional): Discount factor. Defaults to 0.9.
        theta (float, optional): Convergence threshold. Defaults to 1e-6.
        printing (bool, optional): If True, prints out iteration updates. Defaults to False.
        n (int, optional): Iterations to consider in convergence calculation. Defaults to 10.

    Raises:
        ValueError: Discount factor gamma must be between 0 and 1.
        ValueError: Convergence threshold theta must not be negative.

    Returns:
        tuple[dict[Any, float], dict]: Optimal Q table, Optimal policy
            Optimal Q table: Mapping of states to the optimal state value
            Optimal policy: Mapping of states to the best action to take in that state
    """
    # Validate value iteration parameters
    if gamma > 1 or gamma < 0:
        raise ValueError("Discount factor gamma must be between 0 and 1.")

    if theta <= 0:
        raise ValueError("Convergence threshold theta must not be negative.")

    # Get valid state-action pairs for Q-values
    valid_state_actions = {
        (state, action) for state, actions in mdp.actions.items() for action in actions
    }

    # Initialise Q-values as zero for all state-action pairs
    q_table = dict.fromkeys(valid_state_actions, 0.0)

    # Track changes in Q-values for convergence
    delta_history = []

    # Track iterations
    iteration = 0
    while True:
        iteration += 1

        # Sample a valid state-action pair
        s, a = random.choice(list(valid_state_actions))

        # Compute the Q-value update
        old_q_value = q_table[(s, a)]
        q_table[(s, a)] = mdp.rewards.get((s, a), 0) + gamma * sum(
            mdp.probabilities.get((s, a, s_), 0)
            * max(q_table.get((s_, a_), 0) for a_ in mdp.actions.get(s_, []))
            for s_ in mdp.states
        )

        # Obtain the change in the Q-value
        delta_history.append(abs(q_table[(s, a)] - old_q_value))

        # Check for convergence based on average delta over last n iterations
        if len(delta_history) >= n:
            # Calculate the average of last n delta values
            average_delta = sum(delta_history[-n:]) / n

            if printing:
                print(f"Iteration {iteration}, Average delta: {average_delta}")

            if average_delta < theta:
                break

    # Derive policy from Q-values
    policy = extract_policy_from_q_table(mdp=mdp, q_table=q_table)

    return q_table, policy
