"""Implementation of cost functions used in the package (e.g., Softplus, WeightedSum)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from rxn_network.costs.base import CostFunction

if TYPE_CHECKING:
    from rxn_network.reactions.computed import ComputedReaction


class Softplus(CostFunction):
    """The softplus cost function is a smoothed version of the Rectified Linear Unit (ReLU)
    function commonly used in neural networks. It has the property that the output goes
    to 0 as the input goes to negative infinity, but the output approaches a linear
    scaling as the input goes to positive infinity. This is an especially useful mapping
    for determining a cost ranking of a reaction.
    """

    def __init__(
        self,
        temp: float = 300,
        params: list[str] | None = None,
        weights: list[float] | None = None,
    ):
        """
        Args:
            temp: temperature in Kelvin [K]. This serves as a scale factor for the
                output of the function. Higher temperatures -> lower costs. Defaults to
                300 K.
            params: List of data dictionary keys for function parameters used as an
                argument to the softplus function. Defaults to ["energy_per_atom"]
            weights: List of corresponding values by which to weight the
                function parameters. Defaults to [1.0].
        """
        if params is None:
            params = ["energy_per_atom"]
        if weights is None:
            weights = [1.0]
        if temp <= 0:
            raise ValueError("Temperature must be greater than zero!")

        self.temp = temp
        self.params = params
        self.weights = np.array(weights)

    def evaluate(self, rxn: ComputedReaction) -> float:
        """Calculates the cost of reaction based on the initialized parameters and weights.

        Args:
            rxn: A ComputedReaction to evaluate.

        Returns:
            The cost of the reaction.
        """
        values = []
        for p in self.params:
            if rxn.data and p in rxn.data:
                value = rxn.data[p]
            elif hasattr(rxn, p):
                value = getattr(rxn, p)
            else:
                raise ValueError(f"Reaction is missing parameter {p}!")
            values.append(value)

        values_arr = np.array(values)
        total = float(np.dot(values_arr, self.weights))

        return self._softplus(total, self.temp)

    @staticmethod
    def _softplus(x: float, t: float) -> float:
        """The mathematical formula for the softplus function."""
        return np.log(1 + (273 / t) * np.exp(x))

    def __repr__(self):
        return f"Softplus with parameters: {' '.join([f'{k} ({v})' for k, v in zip(self.params, self.weights)])}"


class WeightedSum(CostFunction):
    """This a weighted sum of user-specified parameters.
    cost = param_1*weight_1 + param_2*weight_2 + param_3*weight_3 ...
    """

    def __init__(
        self,
        params: list[str] | None = None,
        weights: list[float] | None = None,
    ):
        """
        Args:
        params: List of data dictionary keys for function parameters used as an
            argument to the weighted summation. Defaults to ["energy_per_atom"].
        weights: List of corresponding values by which to weight the
            function parameters. Defaults to [1.0].
        """
        if params is None:
            params = ["energy_per_atom"]
        if weights is None:
            weights = [1.0]

        self.params = params
        self.weights = np.array(weights)

    def evaluate(self, rxn: ComputedReaction) -> float:
        """Calculates the cost of reaction based on the initialized parameters and weights.

        Args:
            rxn: A ComputedReaction to evaluate.

        Returns:
            The cost of the reaction.
        """
        values = []
        for p in self.params:
            if rxn.data and p in rxn.data:
                value = rxn.data[p]
            elif hasattr(rxn, p):
                value = getattr(rxn, p)
            else:
                raise ValueError(f"Reaction is missing parameter {p}!")
            values.append(value)

        values_arr = np.array(values)
        return float(np.dot(values_arr, self.weights))

    def __repr__(self):
        return f"Weighted sum with parameters: {' '.join([f'{k} ({v})' for k, v in zip(self.params, self.weights)])}"
