# Value Iteration Algorithm

This package provides value iteration algorithms for determining the optimal policy and state values for a given Markov Decision Process (MDP). It includes implementations of both **synchronous** and **asynchronous** value iteration.

- **Synchronous Value Iteration:** Updates all state values simultaneously in each iteration.
- **Asynchronous Value Iteration:** Updates state values individually and asynchronously.

For more details on the underlying algorithms, refer to this [resource](https://artint.info/2e/html2e/ArtInt2e.Ch9.S5.SS2.html#Ch9.F16).


## Installation ⚡
This package can be installed in one of two ways - using `pip` or by cloning the repository.

### Installation via `pip`
To install with `pip`, run the following command in terminal:
```
pip install git+https://github.com/CassandraDurr/value_iteration.git
```

### Installation via Git Clone
To clone the repository and install it locally, run the following commands in terminal:
```
git clone https://github.com/CassandraDurr/value_iteration.git
cd value_iteration
pip install -e .
```

## Requirements 📋
The core installation includes all necessary dependencies for running the package. However, additional dependencies are required for testing and contributing to the package.

To install these additional dependencies, run the following command from the top-level of the repository:
```
pip install -r requirements/requirements-dev.txt
```
These are the development requirements of the package and include tools for testing, code formatting and quality checking.

## Example usage 🎮
An example of how to use this package can be found in the `examples` folder in the Jupyter notebook: `simple_value_iteration_example.ipynb`. This example shows how to install and use the package with a simple example. The functionality demonstrated in this file includes:
- Using **synchronous** value iteration to find the optimal policy and optimal value function using Python inputs.
- Using **asynchronous** value iteration to find the optimal policy and optimal state-action values (Q-table) using Python inputs.
- Running value iteration using CSV input for a more user-friendly experience. This code can also be modified with minimal changes to perform asynchronous value iteration instead of synchronous value iteration.

## Documentation 📚
Each Python file has its own documentation and commentary, but for ease of use, information about each function can be found at website https://cassandradurr.github.io/value_iteration/ hosted on GitHub pages.

## Test 🔨
This package includes a number of unit tests. To run the unit tests navigate to the top level of the repository (where this README is located) and run `pytest` in the terminal. To run the tests and also view a test coverage report, run:
```
coverage run -m pytest && coverage report
```
The repository is currently at 99% coverage.
