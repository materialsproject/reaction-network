"""
Implementation of the softplus cost function.
"""
from typing import List, Optional

import numpy as np

from rxn_network.core.cost_function import CostFunction
from rxn_network.reactions.computed import ComputedReaction


class WeightedSum(CostFunction):
    """
    This a weighted sum of user-specified parameters.
    """

    def __init__(
        self,
        params: Optional[List[str]] = None,
        weights: Optional[List[float]] = None,
    ):
        """
        Args:
            temp: Absolute Temperature [K]. This serves as a scale factor for the output
                of the function. Higher temperatures -> lower costs. Defaults to 300 K.
            params: List of data dictionary keys for function parameters used as an
                argument to the softplus function. Defaults to ["energy_per_atom"]
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
        """
        Calculates the cost of reaction based on the initialized parameters and weights.

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

        return total

    def __repr__(self):
        return (
            "Weighted sum with parameters: "
            f"{' '.join([f'{k} ({v})' for k, v in zip(self.params, self.weights)])}"
        )
