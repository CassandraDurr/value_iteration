"""Module for value iteration."""

import random
from typing import Any

from .mdp import MDP


class ValueIteration:
    """Synchronous value iteration class for Markov Decision Processes (MDPs)."""

    def __init__(
        self, mdp: MDP, gamma: float = 0.9, theta: float = 1e-6, printing: bool = False
    ) -> None:
        """Initialise the value iteration class performing synchronous value iteration.

        Args:
            mdp (MDP): Data class containing states, actions, rewards and probabilities.
            gamma (float, optional): Discount factor. Defaults to 0.9.
            theta (float, optional): Convergence threshold. Defaults to 1e-6.
            printing (bool, optional): If True, prints out iteration updates. Defaults to False.

        Raises:
            ValueError: Parameter for gamma is not between 0 and 1.
            ValueError: Parameter for theta is negative.
        """
        # Validate value iteration parameters
        if gamma > 1 or gamma < 0:
            raise ValueError("Discount factor gamma must be between 0 and 1.")

        if theta <= 0:
            raise ValueError("Convergence threshold theta must not be negative.")

        # Assign parameters
        self.mdp = mdp
        self.gamma = gamma
        self.theta = theta
        self.printing = printing

        # Initialise value function as zero for all states
        self.values = dict.fromkeys(self.mdp.states, 0.0)

    def get_q_value(self, s: Any, a: Any) -> float:
        """Get the Q-value of a state-action pair (s,a) given the value function.

        Args:
            s (Any): State of the MDP.
            a (Any): Action taken at state.
            values (dict[Any, float]): Value function of states.

        Returns:
            float: Q-value.
        """
        return sum(
            self.mdp.probabilities.get((s, a, s_), 0)
            * (self.mdp.rewards.get((s, a, s_), 0) + self.gamma * self.values[s_])
            for s_ in self.mdp.states
        )

    def extract_policy(self) -> dict:
        """Given MDP return the optimal policy.

        Returns:
            dict: Optimal policy - mapping of states to best action.
        """
        # Extract policy
        policy = {}
        for s in self.mdp.states:
            # Initialise best action and value for state
            best_action = None
            best_value = float("-inf")

            for a in self.mdp.actions[s]:
                q_value = self.get_q_value(s=s, a=a)

                if q_value > best_value:
                    best_value = q_value
                    best_action = a

            policy[s] = best_action

        return policy

    def value_iteration(self) -> tuple[dict[Any, float], dict]:
        """Perform synchronous value iteration.

        Returns:
            tuple[dict[Any, float], dict]: Optimal values, Optimal policy
                Optimal values: Mapping of states to the optimal state value
                Optimal policy: Mapping of states to the best action to take in that state
        """
        # Track iterations
        iteration = 0
        while True:
            iteration += 1

            # Copy existing value function
            new_values = self.values.copy()

            # Track changes in the value function for convergence
            delta_values = []

            for s in self.mdp.states:
                # If no valid actions exist for this state, pass
                if s not in self.mdp.actions or not self.mdp.actions[s]:
                    continue

                # Initialise list to store Q values for the state
                q_values = []

                # Get Q for all actions for this state
                for a in self.mdp.actions[s]:
                    q_values.append(self.get_q_value(s=s, a=a))

                # Update value function for the state
                new_values[s] = max(q_values)

                # Calculate absolute change in value function for the state
                delta_values.append(abs(self.values[s] - new_values[s]))

            # Update value function
            self.values = new_values

            if self.printing:
                print(f"Iteration {iteration}, max value change: {max(delta_values)}")

            # Test for convergence
            if max(delta_values) < self.theta:
                break

        # Extract policy
        policy = self.extract_policy()

        return self.values, policy


class AsynchValueIteration(ValueIteration):
    """Asynchronous value iteration class for Markov Decision Processes (MDPs)."""

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        mdp: MDP,
        gamma: float = 0.9,
        theta: float = 1e-6,
        printing: bool = False,
        n: int = 10,
    ):
        """Initialise the value iteration class performing asynchronous value iteration.

        Args:
            mdp (MDP): Data class containing states, actions, rewards and probabilities.
            gamma (float, optional): Discount factor. Defaults to 0.9.
            theta (float, optional): Convergence threshold. Defaults to 1e-6.
            printing (bool, optional): If True, prints out iteration updates. Defaults to False.
            n (int, optional): Convergence parameter for the number of prior iterations to consider.
                Defaults to 10.
        """
        super().__init__(mdp, gamma, theta, printing)
        self.n = n
        # Get valid state-action pairs for Q-values
        self.valid_state_actions = {
            (state, action)
            for state, actions in mdp.actions.items()
            for action in actions
        }

        # Initialise Q-values as zero for all state-action pairs
        self.q_table = dict.fromkeys(self.valid_state_actions, 0.0)

    def get_q_value(self, s: Any, a: Any) -> float:
        """Get the Q-value of a state-action pair (s,a).

        Args:
            s (Any): State of the MDP.
            a (Any): Action taken at state.

        Returns:
            float: Q-value.
        """
        return self.q_table.get((s, a), 0)

    def update_q_table(self, s: Any, a: Any) -> None:
        """Update the Q-value for state s and action a.

        Args:
            s (Any): State of the MDP.
            a (Any): Action taken at state.
        """
        self.q_table[(s, a)] = sum(
            self.mdp.probabilities.get((s, a, s_), 0)
            * (
                self.mdp.rewards.get((s, a, s_), 0)
                + self.gamma
                * max(
                    self.q_table.get((s_, a_), 0) for a_ in self.mdp.actions.get(s_, [])
                )
            )
            for s_ in self.mdp.states
        )

    def value_iteration(self) -> tuple[dict[Any, float], dict]:
        """Perform asynchronous value iteration.

        Returns:
        tuple[dict[Any, float], dict]: Optimal Q table, Optimal policy
            Optimal Q table: Mapping of states to the optimal state value
            Optimal policy: Mapping of states to the best action to take in that state
        """
        # Track changes in Q-values for convergence
        delta_history = []

        # Track iterations
        iteration = 0
        while True:
            iteration += 1

            # Sample a valid state-action pair
            s, a = random.choice(list(self.valid_state_actions))

            # Compute the Q-value update
            old_q_value = self.q_table[(s, a)]
            self.update_q_table(s=s, a=a)

            # Obtain the change in the Q-value
            delta = abs(self.q_table[(s, a)] - old_q_value)
            delta_history.append(delta)

            # Check for convergence based on average delta over last n iterations
            if len(delta_history) >= self.n:
                # Calculate the average of last n delta values
                average_delta = (
                    sum(delta_history[-self.n :])  # noqa: E203 - conflict with black
                    / self.n
                )

                if self.printing:
                    print(f"Iteration {iteration}, Average delta: {average_delta}")

                if average_delta < self.theta:
                    break

        # Derive policy from Q-values
        policy = self.extract_policy()

        return self.q_table, policy
