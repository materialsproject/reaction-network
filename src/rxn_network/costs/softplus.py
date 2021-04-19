" Implementation of the softplus cost function"

from typing import List
import numpy as np

from rxn_network.core import CostFunction
from rxn_network.reactions import ComputedReaction


class Softplus(CostFunction):
    def __init__(
        self,
        temp: float = 300,
        params: List[str] = ["energy_per_atom"],
        weights: List[float] = [1.0],
    ):
        """

        Args:
            temp: Temperature, in Kelvin.
            params: List of data dictionary keys for function parameters used in the
                softplus function.
            weights: List of corresponding values by which to weight the
                function parameters.
        """
        self.temp = temp
        self.params = params
        self.weights = np.array(weights)

    def evaluate(self, rxn: ComputedReaction) -> float:
        """
        Calculates cost of reaction based on the initialized parameters and weights.

        Args:
            rxn: A computed reaction.

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

        values = np.array(values)
        total = np.dot(values, self.weights)

        return self._softplus(total, self.temp)

    @staticmethod
    def _softplus(x, t):
        " The mathematical formula for the softplus function"
        return np.log(1 + (273 / t) * np.exp(x))

    def __repr__(self):
        return (
            f"Softplus with parameters: "
            f"{' '.join([(f'{k} ({v})') for k, v in zip(self.params, self.weights)])}"
        )
