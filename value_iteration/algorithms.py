"""Module for value iteration."""

from typing import Any

from mdp import MDP


def extract_policy_from_values(
    mdp: MDP,
    values: dict[Any, float],
    gamma: float = 0.9,
) -> dict:
    """Given Markov Decision Process (MDP) variables and value function, return the optimal policy.

    Args:
        mdp (MDP): ...
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
        mdp (MDP): ...
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
